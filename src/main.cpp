#include <iostream>
#include <vector>
#include <cstdlib>
#include <tuple>

#define GLFW_INCLUDE_NONE         // to solve conflict of glfw3native and glad
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <glad/glad.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "cl_util.h"
#include "shader.h"

// CL threadblock config
#define THREAD_PER_BLOCK_DIM 16
#define NUM_BLOCKS(n, block_size) (((n) + (block_size) - 1) / (block_size))


// **************** global variables ****************
GLFWwindow * window;
cl::Device device;
cl::Context context;
cl::CommandQueue queue;
cl::Program program;
cl::Kernel kernel, kernelReset;

const float tau = 0.58;
const float uxInit = 0.3, uyInit = 0.06;
const float rhoInit = 1.0;

int winWidth = 0, winHeight = 0;
unsigned int VBO, VAO, EBO;
unsigned int lbmBoundary;
unsigned int lbmBuffer[2][3]; // double buffer, one for read, one for write
cl::ImageGL lbmGLBoundary;
cl::ImageGL lbmGLBuffer[2][3];

// FPS computation
double lastTime = 0.0f;
int nbFrames = 0;
// **************************************************

void framebuffer_size_callback(GLFWwindow * window, int width, int height) {
    glViewport(0, 0, width, height);
}

bool processInput(GLFWwindow *window) {
    // returns whether to reset fluid state
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    else if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        return true;
    return false;
}

std::tuple<double, double> getMouseClickPos(GLFWwindow *window) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        return {xpos, ypos};
    } else
        return {-1, -1};
}

void showFPS(GLFWwindow * window, double interval = 0.1f) {
     double currentTime = glfwGetTime();
     double delta = currentTime - lastTime;
     nbFrames++;
     if (delta >= interval) { 
        double fps = double(nbFrames) / delta;

        std::stringstream ss;
        ss << "LBM" << " [" << fps << " FPS]";

        glfwSetWindowTitle(window, ss.str().c_str());

        nbFrames = 0;
        lastTime = currentTime;
     }
}

void initGL() {
    // glfw: initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    window = glfwCreateWindow(800, 600, "LBM", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(1);
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    // set vsync, make sure the simulation won't go too fast
    glfwSwapInterval(1);
    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(1);
    }
}

bool initFluidState(const char * imagePath) {
    // load image
    int nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char * maskData = stbi_load(imagePath, &winWidth, &winHeight, &nrChannels, 0);
    std::cout << "texture image (HxW):" << winHeight << " x " << winWidth << std::endl;

    // Fill _boundaryData_ with image data from _boundaryBitmap_
    float * boundaryData = new float[winWidth * winHeight * 3];
    if (boundaryData == NULL) {
        std::cout << "Unable to allocate memory!" << std::endl;
        return false;
    }
    for (int y = 0; y < winHeight; y++) {
        for (int x = 0; x < winWidth; x++) {
            int index = y * winWidth + x;
            // Pixels near image margin are set to be boundary 
            if ((x < 2) || (x > (winWidth - 3)) || (y < 2) || (y > (winHeight - 3))) {
                boundaryData[3 * index + 0] = 0.0f;
                boundaryData[3 * index + 1] = 0.0f;
                boundaryData[3 * index + 2] = 0.0f;
            } else {
                // pixels: 0.0 or 1.0
                unsigned char r = maskData[3 * index + 0];
                unsigned char g = maskData[3 * index + 1];
                unsigned char b = maskData[3 * index + 2];
                boundaryData[3 * index + 0] = r / 255.0;
                boundaryData[3 * index + 1] = g / 255.0;
                boundaryData[3 * index + 2] = b / 255.0;
            }
        }
    }
    stbi_image_free(maskData);
    // generate OpenGL texture buffer for Boundary data
    glGenTextures(1, &lbmBoundary);
    glBindTexture(GL_TEXTURE_2D, lbmBoundary);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, winWidth, winHeight, 0, GL_RGB, GL_FLOAT, boundaryData);

    // initialize the data buffers for LBM simulation use
    float * lbmData[4];
    for (int i = 0; i < 4; i++)
    {
        lbmData[i] = new float[winWidth * winHeight * 4];
        if (lbmData[i] == NULL)
            return false;
    }
    // dummy data for initializing the second copy of butter
    memset(lbmData[3], 0, winWidth * winHeight * 4 * sizeof(float));

    // initialize distribution function
    float w[9] = { 4.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 
                   1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };
    float e[9][2] = { { 0,0 }, { 1,0 }, { 0,1 }, { -1,0 }, { 0,-1 }, 
                      { 1,1 }, { -1,1 }, { -1,-1 }, { 1,-1 } };
    // initialize values of f0-f8, rho, ux, uy for each pixel
    for (int y = 0; y < winHeight; y++) {
        for (int x = 0; x < winWidth; x++) {
            float ux = uxInit;
            float uy = uyInit;
            float rho = rhoInit;
            float uu_dot = (ux * ux + uy * uy);
            float f[9];
            for (int i = 0; i < 9; i++) {
                float eu_dot = (e[i][0] * ux + e[i][1] * uy);
                f[i] = w[i] * rho * (1.0f + 3.0f * eu_dot + 4.5f * eu_dot * eu_dot - 1.5f * uu_dot);
            }

            int index = y * winWidth + x;
            //! f1~f4
            lbmData[0][4 * index + 0] = f[1];
            lbmData[0][4 * index + 1] = f[2];
            lbmData[0][4 * index + 2] = f[3];
            lbmData[0][4 * index + 3] = f[4];
            //! f5~f8
            lbmData[1][4 * index + 0] = f[5];
            lbmData[1][4 * index + 1] = f[6];
            lbmData[1][4 * index + 2] = f[7];
            lbmData[1][4 * index + 3] = f[8];
            //! f0, rho, and (ux,uy)
            lbmData[2][4 * index + 0] = f[0];
            lbmData[2][4 * index + 1] = rho;
            lbmData[2][4 * index + 2] = ux;
            lbmData[2][4 * index + 3] = uy;
        }
    }

    // generate OpenGL texture buffer for lbmBuffer[3]
    for (int i = 0; i < 3; i++) {
        glGenTextures(1, &lbmBuffer[0][i]);
        glBindTexture(GL_TEXTURE_2D, lbmBuffer[0][i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        //! set GL_RGBA as internal format for data storage
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, winWidth, winHeight, 0, GL_RGBA, GL_FLOAT, lbmData[i]);
    }
    // generate texture for the double buffer
    for (int i = 0; i < 3; i++) {
        glGenTextures(1, &lbmBuffer[1][i]);
        glBindTexture(GL_TEXTURE_2D, lbmBuffer[1][i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        //! set GL_RGBA as internal format for data storage
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, winWidth, winHeight, 0, GL_RGBA, GL_FLOAT, lbmData[3]); // dummy data
    }

    delete [] boundaryData;
    delete [] lbmData[0];
    delete [] lbmData[1];
    delete [] lbmData[2];
    delete [] lbmData[3];
    return true;
}

void createGLObjs(Shader & renderProgram) {
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
         // positions       // texture coordinates
         1.0f,  1.0f, 0.0f,  1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,  1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // vertex attribute: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // vertex attribute: texture coordinates
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // load and create textures
    // -------------------------
    const char * image_path = "./mask.jpg";
    if (!initFluidState(image_path)) {
        std::cout << "Error: state initialization failed!" << std::endl;
        exit(1);
    }

    // set uniform variables for render.frag
    renderProgram.use();
    glUniform1i(glGetUniformLocation(renderProgram.ID, "boundary_texture"), 0);
    glUniform1i(glGetUniformLocation(renderProgram.ID, "state_texture3"), 1);
}

void initCL() {
    cl_int errCode;

    try {
        std::vector<cl::Device> vDevices;
        cl::Platform plat = getPlatform();
        plat.getDevices(CL_DEVICE_TYPE_GPU, &vDevices);

        for (int i = 0; i < vDevices.size(); i++) {
            if (checkExtnAvailability(vDevices[i])) {
                device = vDevices[i];
                break;
            }
        }

        cl_context_properties cps[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)glfwGetWGLContext(window),
            CL_WGL_HDC_KHR, (cl_context_properties)GetDC(glfwGetWin32Window(window)),
            CL_CONTEXT_PLATFORM, (cl_context_properties)plat(),
            0
        };

        context = cl::Context(device, cps);
        queue = cl::CommandQueue(context, device);
        program = getProgram(context, "lbm.cl", errCode);
        program.build(std::vector<cl::Device>(1, device));
        kernel = cl::Kernel(program, "lbm");
        kernelReset = cl::Kernel(program, "resetFluid");
    } catch(cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        std::string val = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cout << "Log:\n" << val << std::endl;
        exit(1);
    }
}

void CLReferGLTex() {
    cl_int errCode;

    try {
        // boundary tex
        lbmGLBoundary = cl::ImageGL(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 
                                    0, lbmBoundary, &errCode);
        if (errCode != CL_SUCCESS) {
            std::cout << "Failed to create OpenGL texture reference: " << errCode << std::endl;
            exit(1);
        }

        // lbm tex
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                lbmGLBuffer[i][j] = cl::ImageGL(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 
                                                0, lbmBuffer[i][j], &errCode);
                if (errCode != CL_SUCCESS) {
                    std::cout << "Failed to create OpenGL texture reference: " << errCode << std::endl;
                    exit(1);
                }
            }
        }
    } catch(cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        std::string val = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cout << "Log:\n" << val << std::endl;
        exit(1);
    }
}

void CLCompute(int readBufferIdx, float mouse_x, float mouse_y) {
    assert(readBufferIdx == 0 || readBufferIdx == 1);

    cl::Event ev;
    try {
        glFinish();

        std::vector<cl::Memory> objs;
        objs.push_back(lbmGLBoundary);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                objs.push_back(lbmGLBuffer[i][j]);
        
        // acquiring GL textures
        cl_int res = queue.enqueueAcquireGLObjects(&objs, NULL, &ev);
        ev.wait();
        if (res != CL_SUCCESS) {
            std::cout << "Failed acquiring GL object: " << res << std::endl;
            exit(1);
        }
        
        // set kernel args
        kernel.setArg(0, lbmGLBoundary);                        // boundary_tex
        kernel.setArg(1, lbmGLBuffer[readBufferIdx][0]);        // src_state_tex1
        kernel.setArg(2, lbmGLBuffer[readBufferIdx][1]);        // src_state_tex2
        kernel.setArg(3, lbmGLBuffer[readBufferIdx][2]);        // src_state_tex3
        kernel.setArg(4, lbmGLBuffer[1 - readBufferIdx][0]);    // dst_state_tex1
        kernel.setArg(5, lbmGLBuffer[1 - readBufferIdx][1]);    // dst_state_tex2
        kernel.setArg(6, lbmGLBuffer[1 - readBufferIdx][2]);    // dst_state_tex3
        kernel.setArg(7, tau);                                  // tau
        kernel.setArg(8, winWidth);                             // image_size_x
        kernel.setArg(9, winHeight);                            // image_size_y
        kernel.setArg(10, mouse_x);                             // mouse_loc_x
        kernel.setArg(11, (float)winHeight - mouse_y);          // mouse_loc_y

        cl::NDRange blockCfg(THREAD_PER_BLOCK_DIM, THREAD_PER_BLOCK_DIM);
        cl::NDRange gridCfg(blockCfg[0] * NUM_BLOCKS(winWidth, blockCfg[0]), 
                            blockCfg[1] * NUM_BLOCKS(winHeight, blockCfg[1]));

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, gridCfg, blockCfg);

        // release GL textures
        res = queue.enqueueReleaseGLObjects(&objs, NULL, &ev);
        ev.wait();
        if (res != CL_SUCCESS) {
            std::cout << "Failed releasing GL object: " << res << std::endl;
            exit(1);
        }
        queue.finish();
    } catch(cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }
}

void CLResetFluid(int readBufferIdx) {
    assert(readBufferIdx == 0 || readBufferIdx == 1);

    cl::Event ev;
    try {
        glFinish();

        std::vector<cl::Memory> objs;
        for (int j = 0; j < 3; j++)
            objs.push_back(lbmGLBuffer[readBufferIdx][j]);
        
        // acquiring GL textures
        cl_int res = queue.enqueueAcquireGLObjects(&objs, NULL, &ev);
        ev.wait();
        if (res != CL_SUCCESS) {
            std::cout << "Failed acquiring GL object: " << res << std::endl;
            exit(1);
        }
        
        // set kernel args
        kernelReset.setArg(0, lbmGLBuffer[readBufferIdx][0]);        // state_tex1
        kernelReset.setArg(1, lbmGLBuffer[readBufferIdx][1]);        // state_tex2
        kernelReset.setArg(2, lbmGLBuffer[readBufferIdx][2]);        // state_tex3
        kernelReset.setArg(3, rhoInit);                              // init_rho
        kernelReset.setArg(4, winWidth);                             // image_size_x
        kernelReset.setArg(5, winHeight);                            // image_size_y

        cl::NDRange blockCfg(THREAD_PER_BLOCK_DIM, THREAD_PER_BLOCK_DIM);
        cl::NDRange gridCfg(blockCfg[0] * NUM_BLOCKS(winWidth, blockCfg[0]), 
                            blockCfg[1] * NUM_BLOCKS(winHeight, blockCfg[1]));

        queue.enqueueNDRangeKernel(kernelReset, cl::NullRange, gridCfg, blockCfg);

        // release GL textures
        res = queue.enqueueReleaseGLObjects(&objs, NULL, &ev);
        ev.wait();
        if (res != CL_SUCCESS) {
            std::cout << "Failed releasing GL object: " << res << std::endl;
            exit(1);
        }
        queue.finish();
    } catch(cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }
}

void GLRenderFrame(Shader & renderProgram, int readBufferIdx) {
    assert(readBufferIdx == 0 || readBufferIdx == 1);

    glClearColor(199.0 / 255, 237.0 / 255, 204.0 / 255, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    renderProgram.use();
    glBindVertexArray(VAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, lbmBoundary);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, lbmBuffer[1 - readBufferIdx][2]);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

int main() {
    initGL();
    initCL();
    
    Shader renderProgram("./vertex.vert", "./render.frag");
    createGLObjs(renderProgram);
    CLReferGLTex();

    std::cout << "Render loop started ..." << std::endl;
    int readBufferIdx = 0;
    int count = 0;
    while (!glfwWindowShouldClose(window)) {
        bool fReset = processInput(window);
        auto [mouse_x, mouse_y] = getMouseClickPos(window);

        showFPS(window);

        if (fReset)
            CLResetFluid(readBufferIdx);
        CLCompute(readBufferIdx, (float)mouse_x, (float)mouse_y);
        GLRenderFrame(renderProgram, readBufferIdx);
        readBufferIdx = 1 - readBufferIdx;

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();

    std::cout << "Successfully terminated!" << std::endl;
    return 0;
}

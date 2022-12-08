/*
Adopted from https://github.com/9prady9/CLGLInterop

The MIT License (MIT)

Copyright (c) 2015 Pradeep Garigipati

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "cl_util.h"

using namespace cl;

typedef std::vector<Platform>::iterator PlatformIter;

Platform getPlatform(std::string pName, cl_int & error)
{
    Platform ret_val;
    error = 0;
    try {
        // Get available platforms
        std::vector<Platform> platforms;
        Platform::get(&platforms);
        int found = -1;
        for(PlatformIter it=platforms.begin(); it<platforms.end(); ++it) {
            std::string temp = it->getInfo<CL_PLATFORM_NAME>();
            if (temp.find(pName)!=std::string::npos) {
                found = it - platforms.begin();
                std::cout<<"Found platform: "<<temp<<std::endl;
                break;
            }
        }
        if (found==-1) {
            // Going towards + numbers to avoid conflict with OpenCl error codes
            error = +1; // requested platform not found
        } else {
            ret_val = platforms[found];
        }
    } catch(Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
        error = err.err();
    }
    return ret_val;
}

#define FIND_PLATFORM(name)             \
    plat = getPlatform(name, errCode);  \
    if (errCode == CL_SUCCESS)          \
        return plat;

Platform getPlatform()
{
    cl_int errCode;
    Platform plat;

    FIND_PLATFORM(NVIDIA_PLATFORM)
    FIND_PLATFORM(AMD_PLATFORM)
    FIND_PLATFORM(INTEL_PLATFORM)
    FIND_PLATFORM(APPLE_PLATFORM)
    FIND_PLATFORM(MESA_PLATFORM)

    // If no platforms are found
    exit(252);
}

bool checkExtnAvailability(Device & pDevice, const std::string pName)
{
    bool ret_val = true;
    try {
        // find extensions required
        std::string exts = pDevice.getInfo<CL_DEVICE_EXTENSIONS>();
        std::stringstream ss(exts);
        std::string item;
        int found = -1;
        while (std::getline(ss,item,' ')) {
            if (item==pName) {
                found=1;
                break;
            }
        }
        if (found==1) {
            std::cout<<"Found CL_GL_SHARING extension: "<<item<<std::endl;
            ret_val = true;
        } else {
            std::cout<<"CL_GL_SHARING extension not found\n";
            ret_val = false;
        }
    } catch (Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }
    return ret_val;
}

Program getProgram(Context & pContext, const std::string & file, cl_int & error)
{
    Program ret_val;
    error = 0;

    try {
        std::ifstream sourceFile(file.c_str());
        if (!sourceFile.is_open())
            throw Error(1, "cl source not found");
        std::string sourceCode(
                std::istreambuf_iterator<char>(sourceFile),
                (std::istreambuf_iterator<char>()));

        ret_val = Program(pContext, sourceCode);
    } catch(Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
        error = err.err();
    }
    return ret_val;
}

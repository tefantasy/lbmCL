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

#pragma once

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#if defined (__APPLE__) || defined(MACOSX)
static const std::string CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
static const std::string CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif

static const std::string NVIDIA_PLATFORM = "NVIDIA";
static const std::string AMD_PLATFORM = "AMD";
static const std::string MESA_PLATFORM = "Clover";
static const std::string INTEL_PLATFORM = "Intel";
static const std::string APPLE_PLATFORM = "Apple";

cl::Platform getPlatform(std::string pName, cl_int & error);

cl::Platform getPlatform();

bool checkExtnAvailability(cl::Device & pDevice, const std::string pName = CL_GL_SHARING_EXT);

cl::Program getProgram(cl::Context & pContext, const std::string & file, cl_int & error);

#ifndef GL_EXTENSIONS_H
#define GL_EXTENSIONS_H

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#ifdef _WIN32
#define NOMINMAX // Avoiding clashing with std::numeric_limits
#include <windows.h> // Errors in gl.h when not included (VS10)
#endif

#ifdef __APPLE__ // Apple OpenGL framework
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif //__APPLE__

#ifdef _WIN32
#include <GL/glext.h>
#elif defined __APPLE__ // OpenGL framework
#include <OpenGL/glext.h>
#else // Linux
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glext.h>
#endif

#if !defined __APPLE__ // Windows and Linux
typedef void (APIENTRYP PFNGLGETVERTEXATTRIBPOINTERPROC) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid *pointer);
extern PFNGLGENBUFFERSARBPROC glGenBuffers;
extern PFNGLBINDBUFFERPROC glBindBuffer;
extern PFNGLBUFFERDATAPROC glBufferData;
extern PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArray;
extern PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glDisableVertexAttribArray;
extern PFNGLGETVERTEXATTRIBPOINTERPROC glVertexAttribPointer;
extern PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
extern PFNGLUNIFORM2FPROC glUniform2f;
extern PFNGLDELETEBUFFERSPROC glDeleteBuffers;
#endif

// Must be called after GL context has been setup (and before using OpenGL functions imported below).
void init_gl_extensions();

#endif

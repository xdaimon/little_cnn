#pragma once
#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Plot {
	public:
		static void draw ( float x, float y );
		static bool init ( int width, int height );
		static bool init ( int square_width );
		static void display ( );
	private:
		Plot()=delete;
		Plot(Plot const &)=delete;
		Plot(Plot const &&)=delete;
		void operator=(Plot const &)=delete;
		void operator=(Plot const &&)=delete;

		static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void window_size_callback(GLFWwindow* _window, int width, int height);
		static bool compile_shader(char* s, GLuint& sn, GLenum stype);
		static bool link_program(GLuint &pn, GLuint &vs, GLuint &gs, GLuint fs);
};

static GLFWwindow*     window;
static GLuint          vbo;
static GLuint          vao;
static int             wwidth;
static int             wheight;
static GLuint          program;
static GLuint          texture;
static GLint           texture_U;
static GLint           resolution_U;
static std::vector<GLubyte> pixel_store;
static const char* VERTEX_SRC = "       \
#version 330                         \n \
void main(){}";
static const char* GEOMETRY_SRC = "     \
#version 330                         \n \
precision highp float;               \n \
layout(points) in;                   \n \
layout(triangle_strip,max_vertices=6) out;\n \
#define v gl_Position                \n \
void main() {                        \n \
	const vec2 p0 = vec2(-1.,-1.);     \n \
	const vec2 p1 = vec2(-1., 1.);     \n \
	v = vec4(p0,0.,1.); EmitVertex();  \n \
	v = vec4(p1,0.,1.); EmitVertex();  \n \
	v = vec4(-p1,0.,1.); EmitVertex(); \n \
	EndPrimitive();                    \n \
	v = vec4(-p1,0.,1.); EmitVertex(); \n \
	v = vec4(p1,0.,1.); EmitVertex();  \n \
	v = vec4(-p0,0.,1.); EmitVertex(); \n \
	EndPrimitive();}";
static const char* FRAGMENT_SRC = "     \
#version 330                         \n \
uniform sampler2D t; uniform vec2 R; \n \
out vec4 c;                          \n \
void main() {c=texture(t,gl_FragCoord.xy/R);}";

void Plot::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	(void)mods;
	(void)scancode;
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}
void Plot::window_size_callback(GLFWwindow* _window, int width, int height) {
	(void)_window;
	wwidth=width;
	wheight=height;
	glViewport(0, 0, width, height);
}
bool Plot::compile_shader(char* s, GLuint& sn, GLenum stype) {
	sn = glCreateShader(stype);
	glShaderSource(sn, 1, &s, NULL);
	glCompileShader(sn);
	GLint isCompiled = 0;
	glGetShaderiv(sn, GL_COMPILE_STATUS, &isCompiled);
	if(isCompiled == GL_FALSE) {
		GLint maxLength = 0;
		glGetShaderiv(sn, GL_INFO_LOG_LENGTH, &maxLength);
		std::vector<GLchar> errorLog(maxLength);
		glGetShaderInfoLog(sn, maxLength, &maxLength, &errorLog[0]);
		for (auto c : errorLog)
			std::cout << c;
		std::cout << std::endl;
		glDeleteShader(sn);
		return false;
	}
	return true;
}
bool Plot::link_program(GLuint &pn, GLuint &vs, GLuint &gs, GLuint fs) {
	pn = glCreateProgram();
	glAttachShader(pn, gs);
	glAttachShader(pn, fs);
	glAttachShader(pn, vs);
	glLinkProgram(pn);
	GLint isLinked = 0;
	glGetProgramiv(pn, GL_LINK_STATUS, (int *)&isLinked);
	if(isLinked == GL_FALSE) {
		GLint maxLength = 0;
		glGetProgramiv(pn, GL_INFO_LOG_LENGTH, &maxLength);
		std::vector<GLchar> infoLog(maxLength);
		glGetProgramInfoLog(pn, maxLength, &maxLength, &infoLog[0]);
		for (auto c : infoLog)
			std::cout << c;
		std::cout << std::endl;
		glDeleteProgram(pn);
		glDeleteShader(vs);
		glDeleteShader(fs);
		glDeleteShader(gs);
		return false;
	}
	return true;
}
bool Plot::init(int square_width) {
	return init(square_width,square_width);
}
bool Plot::init(int width, int height) {
	static bool init_once = true;
	if (init_once)
		init_once = false;
	else
		return false;
	if (width * height < 3000 * 3000 && width>0 && height>0)
		pixel_store.resize(width*height*4);
	else
		return false;
	wwidth = width;
	wheight = height;
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(wwidth, wheight, "Plot", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSetWindowSizeCallback(window, window_size_callback);
	glfwSwapInterval(1);
	glewExperimental = GL_TRUE;
	glewInit();
	glViewport(0, 0, wwidth, wheight);
	glfwSetKeyCallback(window, key_callback);
	GLuint vs, gs, fs;
	compile_shader((char*)VERTEX_SRC, vs, GL_VERTEX_SHADER);
	compile_shader((char*)GEOMETRY_SRC, gs, GL_GEOMETRY_SHADER);
	compile_shader((char*)FRAGMENT_SRC, fs, GL_FRAGMENT_SHADER);
	link_program(program, vs, gs, fs);
	glClearColor(0., 0., 0., 1.);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, wwidth, wheight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	texture_U = glGetUniformLocation(program, "t");
	resolution_U = glGetUniformLocation(program, "R");
	return true;
}
void Plot::display() {
	bool display_once = true;
	if (display_once) display_once = false;
	else return;
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, wwidth, wheight, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel_store.data());
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(program);
		glUniform1i(texture_U, 0);
		glUniform2f(resolution_U, (GLfloat)wwidth, (GLfloat)wheight);
		glDrawArrays(GL_POINTS, 0, 1);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}
void Plot::draw(float _x, float _y) {
	int x = _x*wwidth;
	int y = _y*wwidth;
	if (x < 0) x = 0;
	if (x >= wwidth) x = wwidth-1;
	if (y < 0) y = 0;
	if (y >= wheight) y = wheight-1;
	pixel_store[4*(y*wwidth+x)+0] = 255;
	pixel_store[4*(y*wwidth+x)+1] = 255;
	pixel_store[4*(y*wwidth+x)+2] = 255;
	pixel_store[4*(y*wwidth+x)+3] = 255;
}

//callbacksPBO.cpp (Rob Farber)

#include <GL/glut.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "types.h"

#define NUM_SHAPES 100

// variables for keyboard control
int animFlag=1;
float animTime=0.0f;
float animInc=0.1f;

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern unsigned int image_width;
extern unsigned int image_height;
extern void moveIn();
extern void moveOut();
extern void moveUp();
extern void moveDown();
extern void moveLeft();
extern void moveRight();

//Mouse
float lastPos[3] = {0.0, 0.0, 0.0};
int curx, cury;
int startX, startY;
bool trackingMouse = false;
bool redrawContinue = false;
bool moveState = false;
bool zoomState = false;
extern sphere_t spheres[NUM_SHAPES];

// The user must create the following routines:
void runCuda();

void display()
{
   // run CUDA kernel
   runCuda();

   // Create a texture from the buffer
   glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);

   // bind texture from PBO
   glBindTexture(GL_TEXTURE_2D, textureID);


   // Note: glTexSubImage2D will perform a format conversion if the
   // buffer is a different format from the texture. We created the
   // texture with format GL_RGBA8. In glTexSubImage2D we specified
   // GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

   // Note: NULL indicates the data resides in device memory
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height,
         GL_RGBA, GL_UNSIGNED_BYTE, NULL);


   // Draw a single Quad with texture coordinates for each vertex.

   glBegin(GL_QUADS);
   glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,0.0f,0.0f);
   glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,0.0f,0.0f);
   glEnd();

   // Don't forget to swap the buffers!
   glutSwapBuffers();

   // if animFlag is true, then indicate the display needs to be redrawn
   if(animFlag) {
      glutPostRedisplay();
      animTime += animInc;
   }
}

//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y)
{
   switch(key) {
   case(27) :
      exit(0);
      break;
   }

   // indicate the display must be redrawn
   glutPostRedisplay();
}

void motion(int x, int y) {
   float curPos[3], dx, dy, dz;

	if(moveState==true){
		curPos[0] = x;
		curPos[1] = y;
		dx = curPos[0] - lastPos[0];
		dy = curPos[1] - lastPos[1];
	

		 for(int s = 0; s < NUM_SHAPES; s++){
		    if (dx) spheres[s].center.x += dx *0.001;
		    if (dy) spheres[s].center.y += dy *0.001;
		 }
		lastPos[0] = curPos[0];
		lastPos[1] = curPos[1];

	}
	else if(zoomState==true){
		curPos[1] = y;
		dy = curPos[1] - lastPos[1];
	
		 for(int s = 0; s < NUM_SHAPES; s++){
		    if (dy) spheres[s].center.z += dy *0.001;
		 }
		lastPos[1] = curPos[1];

	}
	glutPostRedisplay( );

}
void startMotion(long time, int button, int x, int y) {
   trackingMouse = true;
   redrawContinue = false;
   startX = x; startY = y;
   curx = x; cury = y;
}

void stopMotion(long time, int button, int x, int y) {
   
   trackingMouse = false;

   if (startX != x || startY != y)
      redrawContinue = true;
   else {
      redrawContinue = false;
   }
	
}
void mouse(int button, int state, int x, int y)
{
	switch (state) {
      case GLUT_DOWN:
         if (button == GLUT_LEFT_BUTTON) {
            moveState = true;
				lastPos[0] = x;
            lastPos[1] = y;
         }
         else if (button == GLUT_MIDDLE_BUTTON) {
            zoomState = true;
            lastPos[1] = y;
         }
         else startMotion(0, 1, x, y);
         break;
      case GLUT_UP:
         if (button == GLUT_LEFT_BUTTON) {
            moveState = false;
         }
         else if (button == GLUT_MIDDLE_BUTTON) {
            zoomState = false;
         }
         else stopMotion(0, 1, x, y);
         break;
   }

}



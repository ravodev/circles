/*First attempt at the Vanilla Ray Tracer*/

#include <stdio.h>
#include <stdlib.h>
#include "Image.h"
#include "types.h"
#include <time.h>

#define NUM_SHAPES 100

double intercept_sphere(ray_t ray, sphere_t sphere);
color_t lighting(coord_t point, eye_t camera, light_t light, ray_t ray, sphere_t sphere, color_t ambience, sphere_t* spheres);
coord_t cross_prod(coord_t a, coord_t b);
double dot_prod(coord_t a, coord_t b);
coord_t normalize(coord_t a);



int main() {

  //2D Array of Pixels(Colors)
  double xmax = 1023;
  double ymax = 1023;
  double xmin = 0;
  double ymin = 0;
  //color_t pixels[xmax][ymax];
  Image img(xmax+1, ymax+1);
  
  // Set up camera
  eye_t camera;
  camera.eye.x = 0;
  camera.eye.y = 0;
  camera.eye.z = 0;
  
  camera.look.x = 0;
  camera.look.y = 0;
  camera.look.z = -1;
  
  camera.up.x = 0;
  camera.up.y = 1;
  camera.up.z = 0;
  
  // Set up light
  light_t light;
  light.loc.x = 1;
  light.loc.y = 0;
  light.loc.z = 0.5;
  light.color.r = 1;
  light.color.g = 1;
  light.color.b = 1;
  
  // Set up Screen/Near Plane
  screen_t screen;
  screen.xmin = -0.5;
  screen.xmax = 0.5;
  screen.ymin = -0.5;
  screen.ymax = 0.5;
  screen.z = 1; //TODO
  
  // Set up sphere(s)
  sphere_t spheres[NUM_SHAPES];
 
	srand(time(NULL));

	for(int s = 0; s < NUM_SHAPES; s++){
		spheres[s].center.x = ((double)rand() / ((double)RAND_MAX + 1) *2)-1;
  		spheres[s].center.y = ((double)rand() / ((double)RAND_MAX + 1) *2)-1;
  		spheres[s].center.z = 1.5+((double)rand() / ((double)RAND_MAX + 1) *2);
  		spheres[s].radius = .2;
  		spheres[s].color.r = ((double)rand() / ((double)RAND_MAX + 1) );
  		spheres[s].color.g = ((double)rand() / ((double)RAND_MAX + 1) );
  		spheres[s].color.b = ((double)rand() / ((double)RAND_MAX + 1) );
  		spheres[s].spec = .5;
		spheres[s].glos = 5;
  		spheres[s].name = s;
	}
  
  
  ray_t curRay;
  coord_t intercept;
  
  //abient light
  color_t ambience;
  ambience.r = .2;
  ambience.g = .2;
  ambience.b = .2;
  
  
  //int numShapes = 1; //TODO more shapes
  
  // current screen coordinates
  coord_t s;
  
  color_t color;
  
  
  //find coordinates of the screen pixels 
  for (int i=xmin-0.5; i < xmax-0.5; i++) {
    for (int j=ymin-0.5; j < ymax-0.5; j++) {
         //Find x and y values at the screen
         s.x = screen.xmin+(screen.xmax-screen.xmin)*((i+0.5)/abs(xmax-xmin)); //TODO efficentcy+abs
         s.y = screen.ymin+(screen.ymax-screen.ymin)*((j+0.5)/abs(ymax-ymin)); 
         s.z = screen.z;
         
         //convert to proper plane
         coord_t n;
         n.x = camera.eye.x-camera.look.x;
         n.y = camera.eye.y-camera.look.y;
         n.z = camera.eye.z-camera.look.z;
         
         coord_t u = cross_prod(camera.up,n);
         coord_t v = cross_prod(n, u);
         
         u = normalize(u);
         v = normalize(v);
         n = normalize(n);
         
         s.x = camera.eye.x + s.x*u.x + s.y*v.x + s.z*n.x; 
         s.y = camera.eye.y + s.x*u.y + s.y*v.y + s.z*n.y; 
         s.z = camera.eye.z + s.x*u.z + s.y*v.z + s.z*n.z; 
         
         //Define ray
         curRay.dir.x = s.x - camera.eye.x;
         curRay.dir.y = s.y - camera.eye.y;
         curRay.dir.z = s.z - camera.eye.z;
         curRay.start = camera.eye; 
         curRay.t = -1;
         
         float t;
         sphere_t sphere;
         //check which objects intersect with ray
         for(int o = 0; o < NUM_SHAPES; o++){ //TODO more shapes
            t = intercept_sphere(curRay, spheres[o]);
            if ((t > 0 )&&((t < curRay.t) || (curRay.t < 0))){
               curRay.t = t;
               sphere = spheres[o];
            }       
         }
         
         if(curRay.t > 0){
            intercept.x = (curRay.start.x)+curRay.t*(curRay.dir.x);
            intercept.y = (curRay.start.y)+curRay.t*(curRay.dir.y);
            intercept.z = (curRay.start.z)+curRay.t*(curRay.dir.z);
              
         
            

            color = lighting(intercept, camera, light, curRay, sphere, ambience, spheres);
            img.pixel(i, j, color);
            }
      }       
    }
  
  //

  // write the targa file to disk
  img.WriteTga((char *)"awesome.tga", true); 
  // true to scale to max color, false to clamp to 1.0

}

double intercept_sphere(ray_t ray, sphere_t sphere){

   double discrim;
   double t1;
   double t2;
   
   coord_t intercept;
   
   coord_t temp;    //camera - center
   temp.x = ray.start.x - sphere.center.x;
   temp.y = ray.start.y - sphere.center.y;
   temp.z = ray.start.z - sphere.center.z;
   
  
   //find and check discriminant
   discrim=(pow(dot_prod(ray.dir,temp),2)-(dot_prod(ray.dir,ray.dir))*(dot_prod(temp,temp)-pow(sphere.radius,2)));
   
   if(discrim >= 0){
      
      t1 = ((-dot_prod(ray.dir,temp))+(sqrt(discrim)))/(dot_prod(ray.dir,ray.dir));
      t2 = ((-dot_prod(ray.dir,temp))-(sqrt(discrim)))/(dot_prod(ray.dir,ray.dir));
      
      //Find first intercept at t
      if(t1 <= t2){
          return t1;  
      }
      else if (t2 <= t1){
         return t2;
      }
   }
   return -1;
}

color_t lighting(coord_t point, eye_t camera, light_t light, ray_t ray, sphere_t sphere, color_t ambience, sphere_t* spheres){
   coord_t surfNorm;
   coord_t lightNorm;
   coord_t viewNorm;
   coord_t reflectNorm;
   
   ray_t lightRay;
   
   double diffuse;
   double spec;
   color_t color;
   
   //calculate surface normal
   surfNorm.x = point.x - sphere.center.x;
   surfNorm.y = point.y - sphere.center.y;
   surfNorm.z = point.z - sphere.center.z;
   surfNorm = normalize(surfNorm);
   
   //calculate viewing normal
   viewNorm.x = -ray.dir.x;
   viewNorm.y = -ray.dir.y;
   viewNorm.z = -ray.dir.z;
   viewNorm = normalize(viewNorm);
   
   //calculate light normal
   lightNorm.x = light.loc.x-point.x;
   lightNorm.y = light.loc.y-point.y;
   lightNorm.z = light.loc.z-point.z;
   lightNorm = normalize(lightNorm);
   
   //calculate diffuse color
   diffuse = dot_prod(surfNorm,lightNorm);
   if(diffuse > 1) diffuse = 1;
   else if(diffuse < 0) diffuse = 0;
   
   
   //calculate reflection ray normal
   reflectNorm.x = (2*surfNorm.x*diffuse)-lightNorm.x;
   reflectNorm.y = (2*surfNorm.y*diffuse)-lightNorm.y;
   reflectNorm.z = (2*surfNorm.z*diffuse)-lightNorm.z;
   reflectNorm = normalize(reflectNorm);
   
   //calculate specular color
   spec = pow(dot_prod(viewNorm, reflectNorm),sphere.glos);
   
   if (spec > 1) spec = 1;
   else if (spec < 0) spec = 0;
   if(diffuse == 0) spec = 0;  //NOT NEED IN RAY TRACER ?
   
   //check for shadows
   float t = -1;
   bool hit = false;
   lightRay.start = point;
   lightRay.dir = lightNorm;
   
   for(int o = 0; o < NUM_SHAPES; o++){ 
		if (sphere.name != spheres[o].name){
      	t = intercept_sphere(lightRay, spheres[o]);
      	//printf("t: %f  o: %d \n", t, o);
      	if (t > 0){
          	hit = true;
			}
          
      }       
   }
   if(hit){
     spec = 0;
     diffuse = 0;
        
   }
   
   //spec =0;
   //calculate color
   color.r = sphere.color.r*ambience.r + light.color.r*((sphere.color.r*diffuse) + (sphere.spec*spec));
   color.g = sphere.color.g*ambience.g + light.color.g*((sphere.color.g*diffuse) + (sphere.spec*spec));
   color.b = sphere.color.b*ambience.b + light.color.b*((sphere.color.b*diffuse) + (sphere.spec*spec));
   //printf("speck: %f \n", spec);
   
   return color;
   
}

double dot_prod(coord_t a, coord_t b){
   return (a.x)*(b.x)+(a.y)*(b.y)+(a.z)*(b.z);
}

coord_t cross_prod(coord_t a, coord_t b){
   coord_t c;
   c.x = a.y*b.z - a.z*b.y;
   c.y = a.z*b.x - a.x*b.z;
   c.z = a.x*b.y - a.y*b.x;
   
   return c;

}

coord_t normalize(coord_t a){
   double mag = sqrt((a.x)*(a.x)+(a.y)*(a.y)+(a.z)*(a.z));
   a.x = (a.x)/mag;
   a.y = (a.y)/mag;
   a.z = (a.z)/mag;
   return a;
}


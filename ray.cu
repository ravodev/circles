/* CUDA version of the ray tracer program.
 * Combined CPE458/570 Project
 * 
 * Brian Gomberg (bgomberg)
 * Luke Larson (lplarson)
 * Susan Marano (smarano)
 */

#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include <time.h>

#define NUM_SHAPES 100
#define X_MAX 1023
#define Y_MAX 1023

#define BLOCK_SIZE 16

#define LIGHT_X 1
#define LIGHT_Y 0
#define LIGHT_Z 0.5
#define LIGHT_C 1

#define SPHERE_GLOSS 5
#define SPHERE_RADIUS_SQRD .04

#define TIMING

__device__ double intercept_sphere(ray_t ray, sphere_t sphere);
__device__ coord_t cross_prod(coord_t a, coord_t b);
__device__ double dot_prod(coord_t a, coord_t b);
__device__ coord_t normalize(coord_t a);
__device__ float  sqrt2(const float x);
float  sqrt2_host(const float x);

coord_t cross_prod_host(coord_t a, coord_t b);
coord_t normalize_host(coord_t a);


// http://stackoverflow.com/questions/13245258/handle-error-not-found-error-in-cuda
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// in global memory: coord_t point, eye_t camera5, light_t light, sphere_t* spheres
__device__ uchar4 DirectIllumination(coord_t point, light_t light, ray_t ray,
                 sphere_t sphere, sphere_t *spheres);

#ifdef TIMING
__global__ void RayTracer(sphere_t *spheres, uchar4 *output_buffer,
        eye_t camera, light_t light, int *runtime, coord_t n, coord_t u, coord_t v) 
#else
__global__ void RayTracer(sphere_t *spheres, uchar4 *output_buffer,
        eye_t camera, light_t light, coord_t n, coord_t u, coord_t v) 
#endif
{
    int col = blockIdx.x * blockDim.x  + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    coord_t s;

#ifdef TIMING
    clock_t start_time = clock();
#endif

    // Bounds checking
    if (col > X_MAX || row > Y_MAX)
        return;

    //Find x and y values at the screen
    // Coords with respect to eye
    s.x = -0.5+(((double)col)/X_MAX); 
    s.y = -0.5+(((double)row)/Y_MAX);
    s.z = 1;

#ifdef TIMING
    if (row == 200 && col == 200) runtime[0] = (int)clock()-start_time;
#endif
    
    // Convert from eye coordinate system to normal
    s.x = camera.eye.x + s.x*u.x + s.y*v.x + s.z*n.x; 
    s.y = camera.eye.y + s.x*u.y + s.y*v.y + s.z*n.y; 
    s.z = camera.eye.z + s.x*u.z + s.y*v.z + s.z*n.z; 
    
    //Define ray
    ray_t curRay;
    curRay.dir.x = s.x - camera.eye.x;
    curRay.dir.y = s.y - camera.eye.y;
    curRay.dir.z = s.z - camera.eye.z;
    curRay.start = camera.eye; 
    curRay.t = -1;

#ifdef TIMING
    if (row == 200 && col == 200) runtime[1] = (int)clock()-start_time - runtime[0];
#endif
    
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
    
#ifdef TIMING
    if (row == 200 && col == 200) runtime[2] = (int)clock()-start_time - runtime[1];
#endif
    
    // Put inside of DirectIllumination
    // Finds intersection from ray
    coord_t intercept;
    int idx = row*(X_MAX+1)+col;
    if (curRay.t > 0)
    {
       intercept.x = (curRay.start.x)+curRay.t*(curRay.dir.x);
       intercept.y = (curRay.start.y)+curRay.t*(curRay.dir.y);
       intercept.z = (curRay.start.z)+curRay.t*(curRay.dir.z);
        // Change intercept to t
        output_buffer[idx] = DirectIllumination(intercept, light, curRay, sphere, spheres);
    }
    else
    {
        output_buffer[idx].w = 0;
        output_buffer[idx].x = 0;
        output_buffer[idx].y = 0;
        output_buffer[idx].z = 0;
    }

#ifdef TIMING
    if (row == 200 && col == 200) runtime[3] = (int)clock()-start_time - runtime[2];
#endif
}

eye_t camera;
light_t light;
sphere_t spheres[NUM_SHAPES];
coord_t n, v, u;

extern "C" void init_cuda()
{
  // Set up camera
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
  light.loc.x = 1;
  light.loc.y = 0;
  light.loc.z = 0.5;
  light.color.r = 1;
  light.color.g = 1;
  light.color.b = 1;
  
  // Set up sphere(s)
    //srand(time(NULL));

    for(int s = 0; s < NUM_SHAPES; s++){
        spheres[s].center.x = ((double)rand() / ((double)RAND_MAX + 1) *2)-1;
        spheres[s].center.y = ((double)rand() / ((double)RAND_MAX + 1) *2)-1;
        spheres[s].center.z = 1.5+((double)rand() / ((double)RAND_MAX + 1) *2);
        spheres[s].color.r = ((double)rand() / ((double)RAND_MAX + 1) );
        spheres[s].color.g = ((double)rand() / ((double)RAND_MAX + 1) );
        spheres[s].color.b = ((double)rand() / ((double)RAND_MAX + 1) );
        spheres[s].spec = .5;
        spheres[s].name = s;
    }
  
    //convert to proper plane
    n.x = camera.eye.x-camera.look.x;
    n.y = camera.eye.y-camera.look.y;
    n.z = camera.eye.z-camera.look.z;
    
    u = cross_prod_host(camera.up,n);
    v = cross_prod_host(n, u);
    
    u = normalize_host(u);
    v = normalize_host(v);
    n = normalize_host(n);
}

extern "C" void run_cuda(uchar4 *dptr)
{
  // current screen coordinates
  sphere_t *spheresd;
#ifdef TIMING
  int *runtime_d;
  HANDLE_ERROR(cudaMalloc(&runtime_d, sizeof(int)*4));
#endif

  HANDLE_ERROR(cudaMalloc(&spheresd, sizeof(sphere_t) * NUM_SHAPES));
  HANDLE_ERROR(cudaMemcpy(spheresd, spheres, sizeof(sphere_t)*NUM_SHAPES, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(spheresd, spheres, sizeof(sphere_t)*NUM_SHAPES, cudaMemcpyHostToDevice));

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((X_MAX+1+(BLOCK_SIZE-1))/BLOCK_SIZE, (Y_MAX+1+(BLOCK_SIZE)/BLOCK_SIZE));
#ifdef TIMING
  RayTracer<<<gridDim, blockDim>>>(spheresd, dptr, camera, light, runtime_d, n, u, v);
#else
  RayTracer<<<gridDim, blockDim>>>(spheresd, dptr, camera, light, n, u, v);
#endif

#ifdef TIMING
  int runtime[4];
  cudaMemcpy(&runtime, runtime_d, sizeof(int)*4, cudaMemcpyDeviceToHost);

  cudaFree(runtime_d);
#endif
  cudaFree(spheresd);

#ifdef TIMING
  printf("%d %d %d %d\n", runtime[0], runtime[1], runtime[2], runtime[3]);
#endif
}

__device__ double intercept_sphere(ray_t ray, sphere_t sphere) {
   double discrim;
   double t1;
   double t2;
   
   coord_t temp;    //camera - center
   temp.x = ray.start.x - sphere.center.x;
   temp.y = ray.start.y - sphere.center.y;
   temp.z = ray.start.z - sphere.center.z;
   
  
   //find and check discriminant
   double raydir_temp_dot = dot_prod(ray.dir,temp);
   double raydir_raydir_dot = dot_prod(ray.dir,ray.dir);
   double temp_temp_dot = dot_prod(temp,temp);

   discrim=(raydir_temp_dot*raydir_temp_dot-(raydir_raydir_dot)*(temp_temp_dot-SPHERE_RADIUS_SQRD));
   
   if (discrim >= 0) {
      discrim = sqrt2(discrim);
      t1 = ((-raydir_temp_dot)+(discrim))/(raydir_raydir_dot);
      if (t1 < 0) return -1;
      t2 = ((-raydir_temp_dot)-(discrim))/(raydir_raydir_dot);
      if (t2 < 0) return -1;
      return (t1<=t2)?t1:t2;
   }
   return -1;
}


__device__ uchar4 DirectIllumination(coord_t point, light_t light, ray_t ray,
                 sphere_t sphere, sphere_t *spheres){
   coord_t surfNorm;
   coord_t lightNorm;
   coord_t viewNorm;
   coord_t reflectNorm;
   
   ray_t lightRay;
   
   double diffuse;
   double spec = 0;
   uchar4 color;

   //calculate light normal
   lightNorm.x = light.loc.x-point.x;
   lightNorm.y = light.loc.y-point.y;
   lightNorm.z = light.loc.z-point.z;
   
   //check for shadows
   int noHit = 1;
   lightRay.start = point;
   lightRay.dir = lightNorm;
   for(int o = 0; o < NUM_SHAPES; o++){
      if (sphere.name != spheres[o].name && (intercept_sphere(lightRay, spheres[o]) >= 0)) {
         noHit = 0;
         break;
      }
   }

   double r, g, b;
   //calculate color
   r = sphere.color.r*.2;
   g = sphere.color.g*.2;
   b = sphere.color.b*.2;
   if (noHit)
   {
	   //calculate surface normal
	   surfNorm.x = point.x - sphere.center.x;
	   surfNorm.y = point.y - sphere.center.y;
	   surfNorm.z = point.z - sphere.center.z;
	   surfNorm = normalize(surfNorm);

	   //calculate diffuse color
	   diffuse = dot_prod(surfNorm,lightNorm);
	   if (diffuse > 1) diffuse = 1;
	   diffuse *= !(diffuse < 0);

	   if(diffuse > 0) {
	      r += light.color.r*(sphere.color.r*diffuse);
	      g += light.color.g*(sphere.color.g*diffuse);
	      b += light.color.b*(sphere.color.b*diffuse);
	      //calculate viewing normal
	      viewNorm.x = -ray.dir.x;
	      viewNorm.y = -ray.dir.y;
	      viewNorm.z = -ray.dir.z;
	      viewNorm = normalize(viewNorm);

	      //calculate reflection ray normal
	      reflectNorm.x = (2*surfNorm.x*diffuse)-lightNorm.x;
	      reflectNorm.y = (2*surfNorm.y*diffuse)-lightNorm.y;
	      reflectNorm.z = (2*surfNorm.z*diffuse)-lightNorm.z;
	      reflectNorm = normalize(reflectNorm);
	      
	      //calculate specular color
	      spec = pow(dot_prod(viewNorm, reflectNorm),SPHERE_GLOSS);
              if (spec > 1)
	      {
                 //calculate color
	         r += light.color.r*sphere.spec;
	         g += light.color.g*sphere.spec;
	         b += light.color.b*sphere.spec;
	      }
              else if (spec > 0)
	      {
                 //calculate color
	         r += light.color.r*(sphere.spec*spec);
	         g += light.color.g*(sphere.spec*spec);
	         b += light.color.b*(sphere.spec*spec);
	      }
	   }
   }


   r = r>1?1:r;
   g = g>1?1:g;
   b = b>1?1:b;

   color.w = 0;
   color.x = r * 255;
   color.y = g * 255;
   color.z = b * 255;
   
   return color;
}

__device__ double dot_prod(coord_t a, coord_t b){
   return (a.x)*(b.x)+(a.y)*(b.y)+(a.z)*(b.z);
}

__device__ coord_t cross_prod(coord_t a, coord_t b){
   coord_t c;
   c.x = a.y*b.z - a.z*b.y;
   c.y = a.z*b.x - a.x*b.z;
   c.z = a.x*b.y - a.y*b.x;
   
   return c;

}

coord_t cross_prod_host(coord_t a, coord_t b){
   coord_t c;
   c.x = a.y*b.z - a.z*b.y;
   c.y = a.z*b.x - a.x*b.z;
   c.z = a.x*b.y - a.y*b.x;
   
   return c;

}

__device__ coord_t normalize(coord_t a){
   double mag = sqrt2((a.x)*(a.x)+(a.y)*(a.y)+(a.z)*(a.z));
   a.x = (a.x)/mag;
   a.y = (a.y)/mag;
   a.z = (a.z)/mag;
   return a;
}

coord_t normalize_host(coord_t a){
   double mag = sqrt((a.x)*(a.x)+(a.y)*(a.y)+(a.z)*(a.z));
   a.x = (a.x)/mag;
   a.y = (a.y)/mag;
   a.z = (a.z)/mag;
   return a;
}
#define SQRT_MAGIC_F 0x5f3759df 
__device__ float  sqrt2(float x)
{
  const float xhalf = 0.5f*x;
 
  union // get bits for floating value
  {
    float x;
    int i;
  } u;
  u.x = x;
  u.i = SQRT_MAGIC_F - (u.i >> 1);  // gives initial guess y0
  return x*u.x*(1.5f - xhalf*u.x*u.x);// Newton step, repeating increases accuracy 
}

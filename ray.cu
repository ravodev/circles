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

#define BLOCK_SIZE 8

#define LIGHT_X 1
#define LIGHT_Y 0
#define LIGHT_Z 0.5
#define LIGHT_C 1

#define RADIUS_SQUARE 0.04

__device__ double intercept_sphere(ray_t *ray, sphere_t *sphere);
__device__ coord_t cross_prod(coord_t a, coord_t b);
__device__ double dot_prod(coord_t* a, coord_t* b);
__device__ coord_t normalize(coord_t a);
//void doAnimation(Image *image);

__device__ float sqrt7(float x)
{
    unsigned int i = *(unsigned int*) &x; 
    i  += 127 << 23;
    i >>= 1; 
    return *(float*) &i;
}  

// http://stackoverflow.com/questions/13245258/handle-error-not-found-error-in-cuda
static void HandleError(cudaError_t err,
                        const char* file,
                        int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// in global memory: coord_t point, eye_t camera5, light_t light, color_t ambience, sphere_t* spheres
__device__ uchar4 DirectIllumination(coord_t *point, light_t *light, ray_t *ray,
                                     sphere_t *sphere, color_t *ambience, sphere_t* spheres);

__global__ void RayTracer(sphere_t* spheres, uchar4* output_buffer,
                          eye_t camera, light_t light, color_t ambience)
{
    int col = blockIdx.x * blockDim.x  + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    coord_t s;

    // Bounds checking
    if (col > X_MAX || row > Y_MAX)
    {
        return;
    }

    //Find x and y values at the screen
    // Coords with respect to eye
    s.x = -0.5 + (((double)col) / X_MAX);
    s.y = -0.5 + (((double)row) / Y_MAX);
    s.z = 1;

    //convert to proper plane
    coord_t n;
    n.x = camera.eye.x - camera.look.x;
    n.y = camera.eye.y - camera.look.y;
    n.z = camera.eye.z - camera.look.z;

    coord_t u = cross_prod(camera.up, n);
    coord_t v = cross_prod(n, u);

    u = normalize(u);
    v = normalize(v);
    n = normalize(n);

    // Convert from eye coordinate system to normal
    s.x = camera.eye.x + s.x * u.x + s.y * v.x + s.z * n.x;
    s.y = camera.eye.y + s.x * u.y + s.y * v.y + s.z * n.y;
    s.z = camera.eye.z + s.x * u.z + s.y * v.z + s.z * n.z;

    
    ray_t curRay;
    curRay.dir.x = s.x - camera.eye.x;
    curRay.dir.y = s.y - camera.eye.y;
    curRay.dir.z = s.z - camera.eye.z;
    curRay.start = camera.eye;
    curRay.t = -1;

    float t;
    sphere_t *sphere;
    //check which objects intersect with ray
    for (int o = 0; o < NUM_SHAPES; o++)
    {
        t = intercept_sphere(&curRay, &(spheres[o]));
        if ((t > 0) && ((t < curRay.t) || (curRay.t < 0)))
        {
            curRay.t = t;
            sphere = &(spheres[o]);
        }
    }

    // Put inside of DirectIllumination
    // Finds intersection from ray
    coord_t intercept;
    if (curRay.t > 0)
    {
        intercept.x = (curRay.start.x) + curRay.t * (curRay.dir.x);
        intercept.y = (curRay.start.y) + curRay.t * (curRay.dir.y);
        intercept.z = (curRay.start.z) + curRay.t * (curRay.dir.z);
        // Change intercept to t
        output_buffer[row * (X_MAX + 1) + col] = DirectIllumination(&intercept, &light, &curRay, sphere,
                &ambience, spheres);
    }
    else
    {
        output_buffer[row * (X_MAX + 1) + col].w = 0;
        output_buffer[row * (X_MAX + 1) + col].x = 0;
        output_buffer[row * (X_MAX + 1) + col].y = 0;
        output_buffer[row * (X_MAX + 1) + col].z = 0;
    }
}

eye_t camera;
light_t light;
sphere_t spheres[NUM_SHAPES];
color_t ambience;

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
    srand(time(NULL));

    for (int s = 0; s < NUM_SHAPES; s++)
    {
        spheres[s].center.x = ((double)rand() / ((double)RAND_MAX + 1) * 2) - 1;
        spheres[s].center.y = ((double)rand() / ((double)RAND_MAX + 1) * 2) - 1;
        spheres[s].center.z = 1.5 + ((double)rand() / ((double)RAND_MAX + 1) * 2);
        spheres[s].radius = .2;
        spheres[s].color.r = ((double)rand() / ((double)RAND_MAX + 1));
        spheres[s].color.g = ((double)rand() / ((double)RAND_MAX + 1));
        spheres[s].color.b = ((double)rand() / ((double)RAND_MAX + 1));
        spheres[s].spec = .5;
        spheres[s].glos = 5;
        spheres[s].name = s;
    }

    //abient light
    ambience.r = .2;
    ambience.g = .2;
    ambience.b = .2;
}

extern "C" void run_cuda(uchar4* dptr)
{
    // current screen coordinates
    sphere_t* spheresd;

    HANDLE_ERROR(cudaMalloc(&spheresd, sizeof(sphere_t) * NUM_SHAPES));
    HANDLE_ERROR(cudaMemcpy(spheresd, spheres, sizeof(sphere_t)*NUM_SHAPES, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((X_MAX + 1 + (BLOCK_SIZE - 1)) / BLOCK_SIZE, (Y_MAX + 1 + (BLOCK_SIZE) / BLOCK_SIZE));
    RayTracer <<< gridDim, blockDim>>>(spheresd, dptr, camera, light, ambience);
}

__device__ double intercept_sphere(ray_t *ray, sphere_t *sphere)
{
    double discrim;
    double t1;
    double t2;

    coord_t temp;    //camera - center
    temp.x = ray->start.x - sphere->center.x;
    temp.y = ray->start.y - sphere->center.y;
    temp.z = ray->start.z - sphere->center.z;

    // Precompute to optimize for later usage
    double dot_raydir_raydir = dot_prod(&ray->dir, &ray->dir);
    double dot_raydir_temp = dot_prod(&ray->dir, &temp);
    double dot_temp_temp = dot_prod(&temp, &temp) - RADIUS_SQUARE;

    //find and check discriminant
    discrim = (dot_raydir_temp * dot_raydir_temp) - dot_raydir_raydir * dot_temp_temp;

    if (discrim >= 0)
    {
        double dot = -dot_raydir_temp;
        double presqrt7 = sqrt7(discrim);

        t1 = (dot + presqrt7) / dot_raydir_raydir;
        t2 = presqrt7 == 0.0 ? t1 : (dot - presqrt7) / dot_raydir_raydir;

        // Find closer sphere
        if (t1 <= t2)
        {
            return t1;
        }
        else
        {
            return t2;
        }
    }
    return -1;
}

__device__ uchar4 DirectIllumination(coord_t *point, light_t *light, ray_t *ray,
                                     sphere_t *sphere, color_t *ambience, sphere_t *spheres)
{
    coord_t surfNorm;
    coord_t lightNorm;
    coord_t viewNorm;
    coord_t reflectNorm;

    ray_t lightRay;

    double diffuse;
    double spec;
    uchar4 color;

    //calculate surface normal
    surfNorm.x = point->x - sphere->center.x;
    surfNorm.y = point->y - sphere->center.y;
    surfNorm.z = point->z - sphere->center.z;
    surfNorm = normalize(surfNorm);


    //calculate light normal
    lightNorm.x = light->loc.x - point->x;
    lightNorm.y = light->loc.y - point->y;
    lightNorm.z = light->loc.z - point->z;

    //calculate diffuse color
    diffuse = dot_prod(&surfNorm, &lightNorm);

    if (diffuse > 1)
    {
        diffuse = 1;
    }
    diffuse *= !(diffuse < 0);

    if (diffuse > 0)
    {
        //calculate viewing normal
        viewNorm.x = -ray->dir.x;
        viewNorm.y = -ray->dir.y;
        viewNorm.z = -ray->dir.z;
        viewNorm = normalize(viewNorm);

        //calculate reflection ray normal
        reflectNorm.x = (2 * surfNorm.x * diffuse) - lightNorm.x;
        reflectNorm.y = (2 * surfNorm.y * diffuse) - lightNorm.y;
        reflectNorm.z = (2 * surfNorm.z * diffuse) - lightNorm.z;
        reflectNorm = normalize(reflectNorm);

        //calculate specular color
        double dot_viewNorm_reflectNorm = dot_prod(&viewNorm, &reflectNorm);
        spec = dot_viewNorm_reflectNorm * dot_viewNorm_reflectNorm *
               dot_viewNorm_reflectNorm * dot_viewNorm_reflectNorm * dot_viewNorm_reflectNorm;

        spec = (spec > 1) ? 1 : (spec < 0 ? 0 : spec);

        //check for shadows
        float t = -1;
        int noHit = 1;
        lightRay.start = *point;
        lightRay.dir = lightNorm;

        for (int o = 0; o < NUM_SHAPES; o++)
        {
            if (sphere->name != spheres[o].name)
            {
                t = intercept_sphere(&lightRay, &(spheres[o]));
                if (t > 0)
                {
                    noHit = 0;
                    break;
                }
            }
        }
        spec *= noHit;
        diffuse *= noHit;
    }
    else
    {
        spec = 0;
    }

    //calculate color
    double r = sphere->color.r * ambience->r + light->color.r * ((sphere->color.r * diffuse) + (sphere->spec * spec));
    double g = sphere->color.g * ambience->g + light->color.g * ((sphere->color.g * diffuse) + (sphere->spec * spec));
    double b = sphere->color.b * ambience->b + light->color.b * ((sphere->color.b * diffuse) + (sphere->spec * spec));

    r = r > 1 ? 1 : r;
    g = g > 1 ? 1 : g;
    b = b > 1 ? 1 : b;

    color.w = 0;
    color.x = r * 255;
    color.y = g * 255;
    color.z = b * 255;

    return color;
}

__device__ double dot_prod(coord_t* a, coord_t* b)
{
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

__device__ coord_t cross_prod(coord_t a, coord_t b)
{
    coord_t c;
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;

    return c;

}

__device__ coord_t normalize(coord_t a)
{
    double mag = sqrt7((a.x) * (a.x) + (a.y) * (a.y) + (a.z) * (a.z));
    a.x = (a.x) / mag;
    a.y = (a.y) / mag;
    a.z = (a.z) / mag;
    return a;
}


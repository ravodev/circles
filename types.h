#ifndef __TYPES_H__
#define __TYPES_H__

/* Color struct */
typedef struct color_struct {
   double r;
   double g;
   double b;
   double f; // "filter" or "alpha"
} color_t;

/* Coordinate struct */
typedef struct coord_struct {
   double x;
   double y;
   double z;
} coord_t;

/* Light struct */
typedef struct light_struct {
   coord_t loc;
   color_t color;
} light_t;



/* Screen struct */
typedef struct screen_struct {
   double xmin;  
   double xmax;  
   double ymin;   
   double ymax; 
   double z;  
} screen_t;


/* Ray struct */
typedef struct ray_struct {
   coord_t start;
   coord_t dir;
   double t;
} ray_t;

/* Eye struct */
typedef struct eye_struct {
   coord_t eye;
   coord_t look;
   coord_t up;
} eye_t;


/* Sphere struct */
typedef struct sphere_struct {
   coord_t center;
   double radius;
   color_t color;
   double spec;
   double glos;
   int name; 
} sphere_t;

#endif

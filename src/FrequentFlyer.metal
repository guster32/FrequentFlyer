#include <metal_stdlib>
using namespace metal;

float dot2(float3 v) { return dot(v,v); }

// http://iquilezles.org/www/articles/smin/smin.htm
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0 - h);
}

//https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
float udTriangle(float3 p,float3 a,float3 b,float3 c )
{
 float3 ba = b - a;float3 pa = p - a;
 float3 cb = c - b;float3 pb = p - b;
 float3 ac = a - c;float3 pc = p - c;
 float3 nor = cross( ba, ac );

  return sqrt(
    (sign(dot(cross(ba,nor),pa)) +
     sign(dot(cross(cb,nor),pb)) +
     sign(dot(cross(ac,nor),pc))<2.0)
     ?
     min( min(
     dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
     dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
     dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
     :
     dot(nor,pa)*dot(nor,pa)/dot2(nor) );
}

float sdBox(float3 rayPosition, float3 b)
{
    float3 d = abs(rayPosition)-b;
    return length(max(d,float3(0))) + min(max(d.x,max(d.y,d.z)),0.0);
}

//https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
float udQuad( float3 rayPosition, float3 a, float3 b, float3 c, float3 d )
{
  float3 ba = b - a; float3 pa = rayPosition - a;
  float3 cb = c - b; float3 pb = rayPosition - b;
  float3 dc = d - c; float3 pc = rayPosition - c;
  float3 ad = a - d; float3 pd = rayPosition - d;
  float3 nor = cross( ba, ad );

  return sqrt(
    (sign(dot(cross(ba,nor),pa)) +
     sign(dot(cross(cb,nor),pb)) +
     sign(dot(cross(dc,nor),pc)) +
     sign(dot(cross(ad,nor),pd))<3.0)
     ?
     min( min( min(
     dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
     dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
     dot2(dc*clamp(dot(dc,pc)/dot2(dc),0.0,1.0)-pc) ),
     dot2(ad*clamp(dot(ad,pd)/dot2(ad),0.0,1.0)-pd) )
     :
     dot(nor,pa)*dot(nor,pa)/dot2(nor) );
}

float sdGuy(float3 rayPosition, float iTime)
{
    //Distance to a sphere = (distanceFrom sample to center minus radius)
    float sphereSize = 0.25;
    
    //Parabola i.e. bouncing
    float t = fract(iTime);
    float y = 4.0*t*(1.0-t);

    float3 cen = float3(0.0, y, 0.0);
    return length(rayPosition - cen) - sphereSize;
}

// This is the 3D SDF query. Which is the distance to any object in the scene.
float map(float3 rayPosition, float iTime) {
    

    float distanceToSphere = sdGuy(rayPosition, iTime);

    float3 c = float3(0, 0.05, -0.42);
    float3 d = float3(0, 0.0, -0.42);    
    float3 e = float3(0, 0.0, 0.42);
    float3 b = float3(0, 0.3, 0.42);
    float distanceToQuad = udQuad(rayPosition, c, d, e, b);
 
    //Distance to a plane on the x axis is basically how high the rayPosition is on the y axis
    float planeYPosition = -0.25;
    float distanceToPlane = rayPosition.y - (planeYPosition);
    float distanceToBox = sdBox(rayPosition, float3(0.01, 0.01, 0.01));

    return min(min(distanceToBox, distanceToPlane), distanceToQuad);

}

// In order to calculate lighting we need to know the surface normal.
// This can be done by comparing changes in distance to the surface in small increments (derivatives)
// Combining the changes in x,y,z produces vector which is the surface gradient
// The surface gradient is equivalent to the surface normal.
float3 calcNormal(float3 rayPosition, float iTime) {
    float2 e = float2(0.0001, 0.0);
    return normalize(float3(map(rayPosition+e.xyy, iTime)-map(rayPosition-e.xyy, iTime),
        map(rayPosition+e.yxy, iTime)-map(rayPosition-e.yxy, iTime),
        map(rayPosition+e.yyx, iTime)-map(rayPosition-e.yyx, iTime)));
}

//Raymarching algorithm
float castRay(float3 rayOrigin, float3 rayDirection, float iTime) {    
    float t = 0.0;    
    for(int i=0; i<100; i++) {
        float3 rayPosition = rayOrigin + t*rayDirection;

        float distanceToObj = map(rayPosition, iTime);
        //If we are close enough to an object stop the loop
        if(abs(distanceToObj)<0.001) {
            break;
        }
        t += distanceToObj;
        //If we marched too far stop the loop. Far clipping plane
        if(t > 20.0) break;
    }
    if(t > 20.0) t=-1.0;
    return t;
}

kernel void metaltoy(texture2d<float, access::write> output [[texture(0)]],
                    constant float  &iTime [[buffer(0)]],
                    constant float2 &iMouse [[buffer(1)]],
                    uint2 gid [[thread_position_in_grid]]) {
    float2 iResolution = float2(output.get_width(), output.get_height());
    float2 fragCoord = float2(gid);
    float2 pixelCoord = (2.0*fragCoord-iResolution.xy)/iResolution.y;
    pixelCoord.y = -pixelCoord.y;  //Make coordinate system match OpenGL
    
    // animation based on mouse
    float animation = 10*iMouse.x / iResolution.x;

    float fov = 1.5;
    float3 rayOrigin = float3(1.0* sin(animation), 0.05, 1.0*cos(animation));
    float3 cameraTarget = float3(0.0, 0.0, 0.0);

    float3 ww = normalize(cameraTarget - rayOrigin);
    float3 uu = normalize(cross(ww, float3(0, 1,0)));
    float3 vv = normalize(cross(uu, ww));
    
    float3 rayDirection = normalize(pixelCoord.x*uu + pixelCoord.y*vv + fov*ww);

    float3 backgroundColor = float3(0.4, 0.75, 1.0); // Sky
    float3 horizonColor = float3(0.7, 0.75, 0.8); //De-saturated horizon

    //Adds gradient to backgorund based on the raydirection's y
    // Sky is brighter closer to the horizon
    float3 fragColor = backgroundColor - 0.7*rayDirection.y;
    fragColor = mix(fragColor, horizonColor, exp(-10*rayDirection.y));
    
    // CastRay for Camera
    float t = castRay(rayOrigin, rayDirection, iTime);
    //If we hit an object lets do extra work
    if(t>0) {
        float3 rayPosition = rayOrigin + t*rayDirection;
        float3 surfaceNormal = calcNormal(rayPosition, iTime);
        
        //Base material color. (0.2) is the default albeido of grass and generaly used
        //as base material for lighting 
        float3 material = float3(0.18);

        //Keylight (Strongest ligth on the scene) intensity 10 in this case the sun/yellow
        //To calculate the surface color with the light you simply need to
        // dot(product) the surface normal vector with the light vector
        float3 sunDirection = normalize((float3(0.8, 0.4, 0.2))); //Low close to horizon to enhace shadows 
        float3 sunColor = float3(7.0, 4.5, 3.0); //Yellow Keylight intensity 10
        
        //Sky Fill light. Fills the shadows that the key light is not filling
        //Should have an intensity of 1 in this case the sky blue
        float3 skyDirection = float3(0.0, 1.0, 0.0); // Sky is up
        float3 skyColor = float3(0.5, 0.8, 0.9); //Sky is blue

        //Bouce light. light that is reflected off from other surfaces.
        float3 bounceDirection = float3(0.0, -1.0, 0.0); // point up from the bottom
        float3 bounceColor = float3(0.7, 0.3, 0.2); //Sun is yellow and surface is white.
        
        // Self intersection avoidance offset
        float3 offset = surfaceNormal*0.001;
        //cast ray for shadows
        float sunShadow = step(castRay(rayPosition+offset, sunDirection, iTime), 0.0);
        float fillLightOffset = 0.5;
        
        float sunDiffuse = clamp(dot(surfaceNormal, sunDirection), 0.0, 1.0);
        float skyDiffuse = clamp(fillLightOffset+fillLightOffset*dot(surfaceNormal, skyDirection), 0.0, 1.0);
        float bounceDiffuse = clamp(fillLightOffset+fillLightOffset*dot(surfaceNormal, bounceDirection), 0.0, 1.0);
        
        fragColor = material * sunColor * sunDiffuse * sunShadow;
        fragColor += material * skyColor * skyDiffuse;
        fragColor += material * bounceColor * bounceDiffuse;

    }

    // Gramma correction
    fragColor = pow(fragColor, float3(0.4545));

    output.write(float4(fragColor, 1.0), uint2(fragCoord.x, fragCoord.y));
}

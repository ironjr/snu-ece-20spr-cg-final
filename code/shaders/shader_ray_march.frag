#version 330

// Rendering options.
// #define OPTION_RENDER_NORMAL
#define OPTION_RENDER_GLOW
#define OPTION_SHADOW_ON
#define OPTION_SOFT_SHADOW
// Only works if soft shadow is on.
#define OPTION_DETAILED_SHADOW
// #define OPTION_FOG_ON
#define OPTION_RENDER_SUN
#define OPTION_RENDER_CLOUDS
#define OPTION_RENDER_WATER

// 2D noise for the water waves.
// #define OPTION_NOISE2D_SIMPLEX
#define OPTION_NOISE2D_WAVE

// 3D noise for volumetric clouds.
#define OPTION_NOISE3D_GRADIENT
// #define OPTION_NOISE3D_GRADIENT_CUBIC
// #define OPTION_NOISE3D_VORONOISE

// Mathematical constants.
#define PI 3.14159265
#define SQRT_2 1.41421356237
#define RGB_TO_LUMINANCE (vec3(0.2126, 0.7152, 0.0722))

// Rendering parameters.
#define MAX_MARCHING_STEPS 1000
#define EPSILON 1e-6
#define SCATTERED_RAY_BIAS 1e-4
#define ANTIALIASING_JITTER 1e-6
#define ANTIALIASING_SAMPLES 1
// #define ANTIALIASING_JITTER 1e-3
// #define ANTIALIASING_SAMPLES 2
#define MAX_DIST 1000.0
#define MAX_DEPTH 3
#define GLOW_INTENSITY 8
#define VORONOI_AMOUNT 0.9
#define VORONOI_NOISE_LEVEL 0.0

// Illumination options.
#define SUN_BRIGHTNESS 1.5
#define SUN_SHARPNESS 2048.0
#define SUN_COLOR_MULTIPLIER (vec3(1.0, 0.82, 0.68))
#define SUN_CORONA_BRIGHTNESS 0.15
#define SUN_CORONA_SHARPNESS 70.0
#define SUN_CORONA_COLOR_MULTIPLIER (vec3(1.0, 1.0, 0.69))
// #define SUN_SIZE 0.004
#define FOG_BASE_ATTENUATION 0.0001
#define CLOUD_DENSITY_THRESHOLD 0.25
#define CLOUD_EXTINCTION (vec3(0.95, 0.98, 1.0))
#define CLOUD_HEIGHT_MIN 600.0
#define CLOUD_HEIGHT_MAX 700.0
#define CLOUD_MAXIMUM_ATTENUATION 0.98
#define SHADOW_SHARPNESS 10.0

// Scattering options.
#define SCATTER_TYPE_MIRROR        0
#define SCATTER_TYPE_LAMBERTIAN    1
#define SCATTER_TYPE_REFRACTIVE    2
#define SCATTER_TYPE_SPECULAR      3

// Scene configurations.
// TODO make the host code configure the scene.
#define NUM_POINT_LIGHTS 0


struct Material
{
    // Phong shading coefficients
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    float shininess;

    // Reflect / Refract
    vec3 R0; // Schlick approximation
    float ior; // Index of refration (> 1)

    // For refractive material
    vec3 extinction;
    vec3 shadowAttn; // Shadow attenuation

    // TODO use uniform method call
    // 0 : Phong + fresnel + ideal specular (mirror)
    // 1 : Lambertian
    // 2 : Refractive dielectric
    // 3 : Specular
    int scatterType;
};

struct Ray
{
    vec3 origin;
    vec3 direction;
};

struct HitRecord
{
    int s;          // Number of steps before hit
    float t;        // Distance to hit point
    vec3 p;         // Hit point
    vec3 normal;    // Hit point normal
    float vis;      // Visibility related to the closest distance before hit (for soft shadow)
    Material mat;   // Hit point material
};

struct ClosestPoint
{
    float dist;
    Material mat;
};

struct DirectionalLight
{
    // vec3 position;
    vec3 direction;
    vec3 color;
    bool castShadow;
};

struct PointLight
{
    vec3 position;
    vec3 color;
    bool castShadow;
};

// Geometry
struct Sphere
{
    vec3 center;
    float radius;
    Material mat;
};

struct Plane
{
    vec3 normal;
    vec3 p0;
    Material mat;
};

struct Aabb
{
    vec3 bmin;
    vec3 bmax;
    Material mat;
};

struct Box
{
    // Default is a unit box.
    vec3 translation;
    vec3 rotation;
    vec3 scale;
    Material mat;
};

struct Triangle
{
    vec3 v0;
    vec3 v1;
    vec3 v2;
    Material mat;
};

struct Tetrahedron
{
    float sharpness;
    vec3 translation;
    vec3 rotation;
    vec3 scale;
    Material mat;
};

struct Torus
{
    vec2 params;
    vec3 translation;
    vec3 rotation;
    vec3 scale;
    Material mat;
};



uniform float currentTime;
uniform float fovY; // Set this to 45.
uniform vec3 cameraPosition;
uniform mat3 cam2worldRotMatrix;
uniform vec2 resolution;
uniform vec3 ambientLightColor;
uniform DirectionalLight sun;
uniform samplerCube envCubeMap;
uniform int sceneNum;
uniform bool useShadow;
uniform bool renderSun;
uniform bool renderGlow;

// uniform Material material_mesh;

in vec2 TexCoord;

out vec4 FragColor;


Material material_ground = Material(
    vec3(0.3, 0.3, 0.1), // Ka
    vec3(194.0, 186.0, 151.0) / 255.0 * 0.6, // Kd
    vec3(0.4, 0.4, 0.4), // Ks
    88.0, // shininess
    vec3(0.05), // R0
    1.0, // ior
    vec3(0.0), // extinction coefficient
    vec3(0.0), // shadow attenuation
    SCATTER_TYPE_MIRROR
);

Material material_mirror = Material(
    vec3(0.0), // Ka
    vec3(0.03, 0.03, 0.08), // Kd
    vec3(0.0), // Ks
    1.0, // shininess
    vec3(1.0), // R0
    1.0, // ior
    vec3(0.0), // extinction coefficient
    vec3(0.0), // shadow attenuation
    SCATTER_TYPE_MIRROR
);

Material material_lambert = Material(
    vec3(0.0), // Ka
    vec3(0.98), // Kd
    vec3(0.0), // Ks
    1.0, // shininess
    vec3(0.01), // R0
    1.0, // ior
    vec3(0.0), // extinction coefficient
    vec3(0.0), // shadow attenuation
    SCATTER_TYPE_LAMBERTIAN
);

Material material_gold = Material(
    vec3(0.0), // Ka
    vec3(0.8, 0.6, 0.2) * 0.001, // Kd
    vec3(0.4, 0.4, 0.2), // Ks
    200.0, // shininess
    vec3(0.8, 0.6, 0.2), // R0
    1.0, // ior
    vec3(0.0), // extinction coefficient
    vec3(0.0), // shadow attenuation
    SCATTER_TYPE_SPECULAR
);

Material material_dielectric_glass = Material(
    vec3(0.0), // Ka
    vec3(0.0), // Kd
    vec3(0.0), // Ks
    1.0, // shininess
    vec3(0.02), // R0
    1.5, // ior
    vec3(0.80, 0.89, 0.75), // extinction coefficient
    vec3(0.4, 0.7, 0.4), // shadow attenuation
    SCATTER_TYPE_REFRACTIVE
);

Material material_water = Material(
    vec3(0.0), // Ka
    vec3(0.18, 0.23, 0.27), // Kd
    vec3(0.1, 0.13, 0.2), // Ks
    100.0, // shininess
    vec3(0.09), // R0
    1.333, // ior
    vec3(0.80, 0.80, 0.72), // extinction coefficient
    vec3(0.4, 0.4, 0.7), // shadow attenuation
    SCATTER_TYPE_MIRROR
);

Material material_box = Material(
    vec3(0.0), // Ka
    vec3(0.3, 0.3, 0.6), // Kd
    vec3(0.3, 0.3, 0.6), // Ks
    200.0, // shininess
    vec3(0.1), // R0
    1.0, // ior
    vec3(0.0), // extinction coefficient
    vec3(0.0), // shadow attenuation
    SCATTER_TYPE_MIRROR
);


Plane groundPlane = Plane(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 0.0), material_ground);

Sphere spheres[] = Sphere[](
    Sphere(vec3(-2.0, 0.5, -3.0), 1.0, material_lambert),
    Sphere(vec3(2.0, 0.5, -2.0), 1.0, material_lambert),
    Sphere(vec3(2.0, 1.3, 1.0), 0.5, material_lambert),
    Sphere(vec3(0.0), 0.5, material_dielectric_glass),
    Sphere(vec3(0.0), 0.5, material_lambert)
);

Torus tori[] = Torus[](
    Torus(vec2(0.8, 0.3), vec3(0.0, 0.8, 0.0), vec3(0.0), vec3(1.0), material_lambert),
    Torus(vec2(0.7, 0.4), vec3(-2.0, 0.8, 1.5), vec3(0.0), vec3(1.0), material_gold)
);

Box boxes[] = Box[](
    Box(vec3(2.0, 0.25, 1.0), vec3(0.0), vec3(1.5, 0.5, 1.5), material_box)
);

Tetrahedron tetrahedra[] = Tetrahedron[](
    Tetrahedron(1.0, vec3(-2.0, 1.0, 1.5), vec3(0.0), vec3(1.0), material_dielectric_glass)
);

PointLight pointlights[] = PointLight[](
    PointLight(vec3( 3.0, 5.0, 3.0), vec3(1.0, 1.0, 1.0), true),
    PointLight(vec3(-3.0, 5.0, 3.0), vec3(1.0, 1.0, 1.0), false),
    PointLight(vec3(-3.0, 5.0,-3.0), vec3(1.0, 1.0, 1.0), false),
    PointLight(vec3( 3.0, 5.0,-3.0), vec3(1.0, 1.0, 1.0), false)
);



// Declaration of methods.
ClosestPoint sceneSDF(vec4 p);
float heightMap(vec2 coord);


// Utils.
// Maps (-inf, inf) to (0, 1).
vec3 visualize(vec3 v)
{
    return 0.5 + tanh(v) * 0.5;
}

// Returns a varying number between 0 and 1.
// float hash1D(float seed)
// {
//     return fract(cos(seed) * 41415.92653);
// }

float hash1D(vec2 seed)
{
    return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453123);
}

// Returns a random 2D vector.
vec2 hash2D(vec2 seed)
{
    seed = vec2(
        dot(seed, vec2(127.1, 311.7)),
        dot(seed, vec2(269.5, 183.3))
    );
    return fract(sin(seed) * 43758.5453123);
}

// Returns a random 3D vector.
vec3 hash3D(vec3 seed)
{
    seed = vec3(
        dot(seed, vec3(127.1, 311.7, 74.7)),
        dot(seed, vec3(269.5, 183.3, 246.1)),
        dot(seed, vec3(113.5, 271.9, 124.6))
    );
    return fract(sin(seed) * 43758.5453123);
}


vec2 randUnitDisk(vec3 seed)
{
    float rv0 = hash1D(seed.xy);
    float r = rv0 * rv0;
    float phi = hash1D(seed.zx) * 2.0 * PI;
    return r * vec2(cos(phi), sin(phi));
}

vec3 randUnitSphere(vec3 seed)
{
    float rv0 = hash1D(seed.xy);
    float R = rv0 * rv0 * rv0;
    float theta = acos(1.0 - 2.0 * hash1D(seed.yz));
    float phi = hash1D(seed.zx) * 2.0 * PI;
    float cosp = cos(phi);
    float sinp = sin(phi);
    float cost = cos(theta);
    float sint = sin(theta);
    return R * vec3(
        sint * cosp,
        sint * sinp,
        cost
    );
}

// Robert Osada et al., "Shape Distributions", ACM Trans. on Graphics 21(4), 2002.
vec3 randTriangle(vec3 seed, Triangle tri)
{
    float rv0 = hash1D(seed.xy);
    float rv1 = hash1D(seed.yz);
    float sqrtrv0 = sqrt(rv0);
    return tri.v0 + sqrtrv0 * (tri.v1 - tri.v0 + rv1 * (tri.v2 - tri.v1));
}

// Noise functions.
// Simplex noise
// https://www.shadertoy.com/view/Msf3WH
#ifdef OPTION_NOISE2D_SIMPLEX
float noise2D(vec2 seed)
{
    const float CF1 = 0.366025404; // (sqrt(3) - 1) / 2
    const float CF2 = 0.211324865; // (3 - sqrt(3)) / 6

    vec2 i = floor(seed + (seed.x + seed.y) * CF1);
    vec2 a = seed - i + (i.x + i.y) * CF2;
    float m = step(a.y, a.x);
    vec2 o = vec2(m, 1.0 - m);
    vec2 b = a - o + CF2;
    vec2 c = a - 1.0 + 2.0 * CF2;
    vec3 h = max(vec3(0.0), 0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)));
    vec3 n = h * h; // h^2
    n = n * n; // h^4
    n = n * vec3(dot(a, hash2D(i)), dot(b, hash2D(i + o)), dot(c, hash2D(i + 1.0)));
    return (n.x + n.y + n.z) * 70.0;
}
#endif

// Wave noise.
// https://www.shadertoy.com/view/tldSRj
#ifdef OPTION_NOISE2D_WAVE
float noise2D(vec2 seed)
{
    const float CF = 3.1415926536;

    vec2 integral = floor(seed);
    vec2 fraction = fract(seed);
    fraction = fraction * fraction * (3.0 - 2.0 * fraction);
    vec2 e = vec2(1.0, 0.0);
    return mix(
        mix(
            sin(CF * dot(seed, hash2D(integral))),
            sin(CF * dot(seed, hash2D(integral + e.xy))),
            fraction.x
        ),
        mix(
            sin(CF * dot(seed, hash2D(integral + e.yx))),
            sin(CF * dot(seed, hash2D(integral + e.xx))),
            fraction.x
        ),
        fraction.y
    );
}
#endif

// Gradient noise with gradient.
// https://www.shadertoy.com/view/4dffRH
#ifdef OPTION_NOISE3D_GRADIENT
vec4 noiseG3D(vec3 seed)
{
    // Grid.
    vec3 i = floor(seed);
    vec3 f = fract(seed);
    vec3 f2 = f * f;
    vec3 f3 = f2 * f;

    // Interpolations: Quintic interpolation removes griddy artifacts.
#ifdef OPTION_NOISE3D_GRADIENT_CUBIC
    vec3 u = 3.0 * f2 - 2.0 * f3;
    vec3 du = 6.0 * (f - f2);
#else
    vec3 u = f3 * (6.0 * f2 - 15.0 * f + 10.0);
    vec3 du = 30.0 * f2 * (f2 - 2.0 * f + 1.0);
#endif

    // Gradients.
    vec2 e = vec2(0.0, 1.0);
    vec3 g0 = hash3D(i);
    vec3 g1 = hash3D(i + e.yxx);
    vec3 g2 = hash3D(i + e.xyx);
    vec3 g3 = hash3D(i + e.yyx);
    vec3 g4 = hash3D(i + e.xxy);
    vec3 g5 = hash3D(i + e.yxy);
    vec3 g6 = hash3D(i + e.xyy);
    vec3 g7 = hash3D(i + e.yyy);

    // Projections.
    float v0 = dot(g0, f);
    float v1 = dot(g1, f - e.yxx);
    float v2 = dot(g2, f - e.xyx);
    float v3 = dot(g3, f - e.yyx);
    float v4 = dot(g4, f - e.xxy);
    float v5 = dot(g5, f - e.yxy);
    float v6 = dot(g6, f - e.xyy);
    float v7 = dot(g7, f - e.yyy);

    // Return the interpolation.
    vec3 uu = u * u.yzx;
    float uuu = u.x * uu.y;
    vec3 vDiff1 = vec3(v1, v2, v4) - v0;
    vec3 vDiff2 = vec3(v0 - v1 - v2 + v3, v0 - v2 - v4 + v6, v0 - v1 - v4 + v5);
    float vDiff3 = -v0 + v1 + v2 - v3 + v4 - v5 - v6 + v7;
    return vec4(
        // Value interpolation (x).
        v0 + dot(u, vDiff1) + dot(uu, vDiff2) + uuu * vDiff3,

        // Gradient interpolation (yzw).
        g0 + u.x * (g1 - g0) + u.y * (g2 - g0) + u.z * (g4 - g0) +
        + uu.x * (g0 - g1 - g2 + g3) + uu.y * (g0 - g2 - g4 + g6) + uu.z * (g0 - g1 - g4 + g5)
        + uuu * (-g0 + g1 + g2 - g3 + g4 - g5 - g6 + g7)
        + du * (vDiff1 + u.yzx * vDiff2 + u.zxy * vDiff2.zxy + uu.yzx * vDiff3)
    );
}

// Gradient noise returns only the value.
float noise3D(vec3 seed)
{
    // Grid.
    vec3 i = floor(seed);
    vec3 f = fract(seed);
    vec3 f2 = f * f;
    vec3 f3 = f2 * f;

    // Interpolations: Quintic interpolation removes griddy artifacts.
#ifdef OPTION_NOISE3D_GRADIENT_CUBIC
    vec3 u = 3.0 * f2 - 2.0 * f3;
#else
    vec3 u = f3 * (6.0 * f2 - 15.0 * f + 10.0);
#endif

    // Gradients.
    vec2 e = vec2(0.0, 1.0);
    return mix(
        mix(
            mix(
                dot(hash3D(i), f),
                dot(hash3D(i + e.yxx), f - e.yxx),
                u.x
            ),
            mix(
                dot(hash3D(i + e.xyx), f - e.xyx),
                dot(hash3D(i + e.yyx), f - e.yyx),
                u.x
            ),
            u.y
        ),
        mix(
            mix(
                dot(hash3D(i + e.xxy), f - e.xxy),
                dot(hash3D(i + e.yxy), f - e.yxy),
                u.x
            ),
            mix(
                dot(hash3D(i + e.xyy), f - e.xyy),
                dot(hash3D(i + e.yyy), f - e.yyy),
                u.x
            ),
            u.y
        ),
        u.z
    );
}
#else

// TODO fix this!!!
#ifdef OPTION_NOISE3D_VORONOISE
// Voronoi tesselation with noise.
float noise3D(vec3 seed, float voronoiAmount, float noiseAmount)
{
    vec3 integral = floor(seed);
    vec3 fraction = fract(seed);
    float noisePower = 1.0 + 63.0 * pow(1.0 - noiseAmount, 4.0);

    float value = 0.0;
    float totalWeight = 0.0;
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            for (int k = -1; k <= 1; ++k)
            {
                ivec3 grid = ivec3(i, j, k);
                // [-1, 1]
                vec3 offsets = 2.0 * hash3D(seed + grid) * vec3(voronoiAmount) - 1.0;
                vec3 voronoi = grid - fraction + offsets;
                float dist = length(voronoi);
                float weight = pow(1.0 - smoothstep(0.0, SQRT_2, dist), noisePower);
                value += weight * hash1D(seed.xz + seed.zy + dot(grid, vec3(1.0, 1.3, 1.7)));
                totalWeight += weight;
            }
        }
    }
    return value / totalWeight;
}
#endif

#endif


// Transforms.
mat3 rotX(float rad)
{
    float c = cos(rad);
    float s = sin(rad);
    return mat3(
        vec3(1, 0,  0),
        vec3(0, c, -s),
        vec3(0, s,  c)
    );
}

mat3 rotY(float rad)
{
    float c = cos(rad);
    float s = sin(rad);
    return mat3(
        vec3( c, 0, s),
        vec3( 0, 1, 0),
        vec3(-s, 0, c)
    );
}

mat3 rotZ(float rad)
{
    float c = cos(rad);
    float s = sin(rad);
    return mat3(
        vec3(c, -s, 0),
        vec3(s,  c, 0),
        vec3(0,  0, 1)
    );
}


float luminance(vec3 rgb)
{
    return dot(rgb, RGB_TO_LUMINANCE);
}

// Schlick's approximation of Fresnel effect.
float schlick(float cosine, float ior)
{
    float r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;

    float cosrev = 1.0 - cosine;
    float cosrev2 = cosrev * cosrev;
    float cosrev4 = cosrev2 * cosrev2;
    float cosrev5 = cosrev4 * cosrev;
    return r0 + (1.0 - r0) * cosrev5;
}

vec3 schlick(float cosine, vec3 r0)
{
    float cosrev = 1.0 - cosine;
    float cosrev2 = cosrev * cosrev;
    float cosrev4 = cosrev2 * cosrev2;
    float cosrev5 = cosrev4 * cosrev;
    return r0 + (vec3(1.0) - r0) * cosrev5;
}

// Floating point operations.
// Quadratic smooth minimum.
// https://www.iquilezles.org/www/articles/smin/smin.htm
float smin(float a, float b, float k)
{
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// Cubic smooth minimum.
// https://www.iquilezles.org/www/articles/smin/smin.htm
// float smin(float a, float b, float k)
// {
//     float h = max(k - abs(a - b), 0.0) / k;
//     return min(a, b) - h * h * h * k * 0.1666666666;
// }

float smax(float a, float b, float k)
{
    float h = max(k - abs(a - b), 0.0) / k;
    return max(a, b) + h * h * k * 0.25;
}

Material mixMaterial(Material aMat, Material bMat, float h)
{
    Material resMat;
    resMat.Ka = mix(aMat.Ka, bMat.Ka, h);
    resMat.Kd = mix(aMat.Kd, bMat.Kd, h);
    resMat.Ks = mix(aMat.Ks, bMat.Ks, h);
    resMat.shininess = mix(aMat.shininess, bMat.shininess, h);
    resMat.R0 = mix(aMat.R0, bMat.R0, h);
    resMat.ior = mix(aMat.ior, bMat.ior, h);
    resMat.extinction = mix(aMat.extinction, bMat.extinction, h);
    resMat.shadowAttn = mix(aMat.shadowAttn, bMat.shadowAttn, h);
    resMat.scatterType = (h < 0.5) ? aMat.scatterType : bMat.scatterType;
    return resMat;
}


// Generate fog.
vec3 applyFog(vec3 rgb, float base, float dist, inout vec3 attenuation)
{
    const vec3 FOG_COLOR = vec3(0.5, 0.56, 0.71);

    float rgbRatio = exp(-dist * base);
    attenuation *= rgbRatio; 
    return rgb * rgbRatio + FOG_COLOR * (1.0 - rgbRatio);
}

// Fog with sun.
vec3 applyFog(vec3 rgb, float base, float dist, float sunCosine, inout vec3 attenuation)
{
    const vec3 FOG_COLOR = vec3(0.5, 0.56, 0.71);
    const vec3 FOG_COLOR_SUN = vec3(1.0, 0.88, 0.67);
    const float FOG_SUN_SHARPNESS = 8.0;

    float sunRatio = pow(max(0.0, sunCosine), FOG_SUN_SHARPNESS);
    float rgbRatio = exp(-dist * base);
    attenuation *= rgbRatio; 
    return rgb * rgbRatio + mix(FOG_COLOR, FOG_COLOR_SUN, sunRatio) * (1.0 - rgbRatio);
}


// A faster formula to find the gradient/normal direction of the SDF.
// (the w component is the average signed distance)
// http://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
// vec3 getNormal(vec4 p, float dx)
// {
//     const vec3 k = vec3(1.0, -1.0, 0.0);
//     return normalize(
//         k.xyy * sceneSDF(p + k.xyyz * dx).dist +
//         k.yyx * sceneSDF(p + k.yyxz * dx).dist +
//         k.yxy * sceneSDF(p + k.yxyz * dx).dist +
//         k.xxx * sceneSDF(p + k.xxxz * dx).dist
//     );
// }

// A standard fomula for finding the normal direction.
vec3 getNormal(vec4 p, float dx) {
    const vec2 k = vec2(1.0, 0);
    return normalize(vec3(
        sceneSDF(p + k.xyyy * dx).dist - sceneSDF(p - k.xyyy * dx).dist,
        sceneSDF(p + k.yxyy * dx).dist - sceneSDF(p - k.yxyy * dx).dist,
        sceneSDF(p + k.yyxy * dx).dist - sceneSDF(p - k.yyxy * dx).dist
    ));
}

vec3 heightMapNormal(vec2 coord, float dx, float smoothness)
{
    const vec2 k = vec2(1.0, 0);
    return normalize(vec3(
        heightMap(coord - k.xy * dx) - heightMap(coord + k.xy * dx),
        2.0 * smoothness * dx,
        heightMap(coord - k.yx * dx) - heightMap(coord + k.yx * dx)
    ));
}

// Fractional Brownian motion given a noise function.
float fBM(vec3 p, float H, int numOctaves)
{
    // Procedurally accumulate the noise of different scales.
    float G = exp2(-H);
    float acc = 0.0;
    float scale = 1.0;
    float amp = 1.0;
    for (int i = 0; i < numOctaves; ++i)
    {
#ifdef OPTION_NOISE3D_VORONOISE
        acc += amp * noise3D(amp * p, VORONOI_AMOUNT, VORONOI_NOISE_LEVEL);
#else
        acc += amp * noise3D(amp * p);
#endif
        scale *= 2.0;
        amp *= G;
    }
    return acc;
}

// 2D fractional Brownian motion.
float fBM(vec2 p, float H, int numOctaves)
{
    // Procedurally accumulate the noise of different scales.
    float G = exp2(-H);
    float acc = 0.0;
    float scale = 1.0;
    float amp = 1.0;
    for (int i = 0; i < numOctaves; ++i)
    {
        acc += amp * noise2D(amp * p);
        scale *= 2.0;
        amp *= G;
    }
    return acc;
}

// 3D fractional Brownian motion with transformation.
// float fBM(vec3 p, mat3 tf, float H, int numOctaves)
// {
//     // Procedurally accumulate the noise of different scales.
//     float G = exp2(-H);
//     float acc = 0.0;
//     float scale = 1.0;
//     float amp = 1.0;
//     for (int i = 0; i < numOctaves; ++i)
//     {
//         acc += amp * noise3D(amp * p);
//         scale *= 2.0;
//         amp *= G;
//         p = tf * p;
//     }
//     return acc;
// }

// Spatio-temporal fractional Brownian motion with transformation.
float fBM(vec3 p, mat3 tf, float H, int numOctaves)
{
    // Procedurally accumulate the noise of different scales.
    float G = exp2(-H);
    float acc = 0.0;
    float scale = 1.0;
    float amp = 1.0;
    for (int i = 0; i < numOctaves; ++i)
    {
        acc -= amp * abs(sin(noise2D(amp * (p.xy + p.z))));
        scale *= 2.0;
        amp *= G;
        p = tf * p;
    }
    return acc;
}

// Signed distance functions (SDFs) of geometric primitives.
float planeSDF(vec4 p)
{
    return p.y / p.w;
}

float sphereSDF(vec4 p, float r)
{
	return (length(p.xyz) - r) / p.w;
}

float boxSDF(vec4 p, vec3 s)
{
	vec3 a = abs(p.xyz) - s * 0.5;
	return (min(max(max(a.x, a.y), a.z), 0.0) + length(max(a, 0.0))) / p.w;
}

float tetrahedronSDF(vec4 p, float r)
{
	float md = max(max(-p.x - p.y - p.z, p.x + p.y - p.z),
				max(-p.x + p.y + p.z, p.x - p.y + p.z));
	return (md - r) / (p.w * sqrt(3.0));
}

float capsuleSDF(vec4 p, float h, float r)
{
	p.y -= clamp(p.y, -h, h);
	return (length(p.xyz) - r) / p.w;
}

float ellipsoidSDF(vec4 p, vec3 r) // approximated
{
    vec3 p3 = p.xyz / p.w;
    float k0 = length(p3 / r);
    float k1 = length(p3 / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float torusSDF(vec4 p, vec2 t)
{
    vec3 p3 = p.xyz / p.w;
    return length(vec2(length(p3.xz) - t.x, p3.y)) - t.y;
}

float cappedTorusSDF(vec4 p, vec2 sc, float ra, float rb)
{
    vec3 p3 = p.xyz / p.w;
    p3.x = abs(p3.x);
    float k = (sc.y * p3.x > sc.x * p3.y) ? dot(p3.xy, sc) : length(p3.xy);
    return sqrt(dot(p3, p3) + ra * ra - 2.0 * ra * k) - rb;
}

float roundBoxSDF(vec4 p, vec3 b, float r)
{
    vec3 p3 = p.xyz / p.w;
    vec3 d = abs(p3) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0)) - r;
}

float crossSDF(vec4 p, vec3 b)
{
    float inf = 1.0 / 0.0;
    float boxX = boxSDF(p.xyzw, vec3(inf, b.y, b.z));
    float boxY = boxSDF(p.yzxw, vec3(b.x, inf, b.z));
    float boxZ = boxSDF(p.zxyw, vec3(b.x, b.y, inf));
    // Union
    return min(min(boxX, boxY), boxZ);
}
    


// Operations on SDFs.
float opIntersection(float distA, float distB)
{
    return max(distA, distB);
}

float opUnion(float distA, float distB)
{
    return min(distA, distB);
}

float opSubtraction(float distA, float distB)
{
    return max(distA, -distB);
}

// https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
vec4 opTwist(vec4 p, float k)
{
    vec3 p3 = p.xyz / p.w;
    float c = cos(k * p3.y);
    float s = sin(k * p3.y);
    mat2 m = mat2(c, -s, s, c);
    vec4 q = vec4(vec3(m * p3.xz, p3.y) * p.w, p.w);
    return q;
}

vec4 opRepeat(vec4 p, vec3 grid)
{
    vec3 p3 = p.xyz / p.w;
    return vec4((mod(p3 + 0.5 * grid, grid) - 0.5 * grid) * p.w, p.w);
}

// TODO add parameters
vec4 opRepeatWave(vec4 p, vec3 grid)
{
    vec3 p3 = p.xyz / p.w;
    vec3 idx = (p3 + 0.5 * grid) / grid;
    vec3 waveOffset = vec3(
        0.0,
        sin(2.0 * currentTime + int(idx.x) * 0.3333) * cos(2.0 * currentTime + int(idx.z) * 0.3333),
        0.0
    );
    vec3 gridOffset = mod(p3 + 0.5 * grid, grid) - 0.5 * grid;
    return vec4((waveOffset + gridOffset) * p.w, p.w);
}

float opRound(float dist, float radius)
{
    return dist - radius;
}

vec4 opTranslate(vec4 p, vec3 off)
{
    return p - vec4(off * p.w, 0.0);
}

vec4 opRotate(vec4 p, mat3 rot)
{
    // Rotation matrices are orthogonal.
    return vec4(transpose(rot) * p.xyz, p.w);
}

vec4 opTransform(vec4 p, mat4 m)
{
    return inverse(m) * p;
}

vec4 opSymX(vec4 p)
{
    p.x = abs(p.x);
    return p;
}

vec4 opSymY(vec4 p)
{
    p.y = abs(p.y);
    return p;
}

vec4 opSymZ(vec4 p)
{
    p.z = abs(p.z);
    return p;
}

vec4 opSymXY(vec4 p)
{
    p.xy = abs(p.xy);
    return p;
}

vec4 opSymYZ(vec4 p)
{
    p.yz = abs(p.yz);
    return p;
}

vec4 opSymZX(vec4 p)
{
    p.zx = abs(p.zx);
    return p;
}

vec4 opFold(vec4 p, vec3 normal)
{
    return vec4(p.xyz - 2.0 * min(0.0, dot(p.xyz, normal)), p.w);
}

ClosestPoint opIntersection(ClosestPoint cpA, ClosestPoint cpB)
{
    return (cpA.dist > cpB.dist) ? cpA : cpB;
}

ClosestPoint opUnion(ClosestPoint cpA, ClosestPoint cpB)
{
    return (cpA.dist < cpB.dist) ? cpA : cpB;
}

ClosestPoint opSubtraction(ClosestPoint cpA, ClosestPoint cpB)
{
    ClosestPoint dummy;
    dummy.dist = -(cpB.dist);
    return (cpA.dist > dummy.dist) ? cpA : dummy;
}

ClosestPoint opSmoothIntersection(ClosestPoint cpA, ClosestPoint cpB, float k)
{
    // float h = max(k - abs(a - b), 0.0) / k;
    // float d = max(a, b) - h * h * k * 0.25;
    float h = clamp(k - (cpB.dist - cpA.dist), 0.0, 2.0 * k) * 0.5 / k;
    float d = mix(cpB.dist, cpA.dist, h) + k * h * (1.0 - h);
    return ClosestPoint(d, mixMaterial(cpB.mat, cpA.mat, h));
}

ClosestPoint opSmoothUnion(ClosestPoint cpA, ClosestPoint cpB, float k)
{
    // float h = max(k - abs(a - b), 0.0) / k;
    // float d = min(a, b) - h * h * k * 0.25;
    float h = clamp(k + (cpB.dist - cpA.dist), 0.0, 2.0 * k) * 0.5 / k;
    float d = mix(cpB.dist, cpA.dist, h) - k * h * (1.0 - h);
    return ClosestPoint(d, mixMaterial(cpB.mat, cpA.mat, h));
}

ClosestPoint opSmoothSubtraction(ClosestPoint cpA, ClosestPoint cpB, float k)
{
    float h = max(k - abs(cpA.dist + cpB.dist), 0.0) / k;
    float d = max(cpA.dist, -cpB.dist) + h * h * k * 0.25;
    return ClosestPoint(d, cpA.mat);
}


// Fractal SDFs
// TODO Fix this!!!!!!!!!!
vec4 sierpinskiFold(vec4 p, int numStep)
{
    const float SCALE_PER_STEP = 2.0;

    vec3 off = vec3(1.0);
    float scale = 1.0;
    for (int i = 0; i < numStep; ++i)
    {
        p.xy -= min(p.x + p.y, 0.0);
        p.yz -= min(p.y + p.z, 0.0);
        p.zx -= min(p.z + p.x, 0.0);
        p.xyz = (p.xyz - off) * SCALE_PER_STEP + off;
        scale *= SCALE_PER_STEP;
    }
    return p;
}

float mengerSpongeSDF(vec4 p, int numStep)
{
    const float SCALE_PER_STEP = 3.0;

    float scale = 1.0;
    float dist = boxSDF(p, vec3(scale));
    float crossDist;
    vec4 crossPoint;
    for (int i = 0; i < numStep; ++i)
    {
        crossPoint = opRepeat(p, vec3(scale));
        scale /= SCALE_PER_STEP;
        crossDist = crossSDF(crossPoint, vec3(scale));
        dist = opSubtraction(dist, crossDist);
    }
    return dist;
}

// TODO Fix this!!!!!!!!!!
float mandelburbSDF(vec4 p, int numStep)
{
    const float THRESHOLD = 256.0;

    vec3 p3 = p.xyz / p.w;
    float dr = 1.0;
    float r = 0.0;
    for (int i = 0; i < numStep; ++i)
    {
        // Get polar coordinates.
        r = length(p3);
        if (r > THRESHOLD)
            break;
        float theta = acos(p3.z / r);
        float phi = atan(p3.y, p3.x);

        // Apply transformations.
        float r2 = r * r; // r^2
        float r4 = r2 * r2; // r^4
        r = r4 * r4; // r^8
        dr = 8.0 * r4 * r2 * r * dr + 1.0;
        theta *= 8.0;
        phi *= 8.0;

        // Back to Cartesian coordinates.
        float st = sin(theta);
        float ct = cos(theta);
        float sp = sin(phi);
        float cp = cos(phi);
        p3 = r * vec3(st * sp, ct, st * cp) + p.xyz;
    }
    return 0.5 * log(r) * r / dr;
}


// SDF of the scene.
ClosestPoint sceneSDF(vec4 p)
{
    ClosestPoint resCP;
    resCP.dist = 1.0 / 0.0;

    if (sceneNum == 1 || sceneNum == 2)
    {
        // A single sphere.
        {
        vec4 spherePoint = p - vec4(spheres[0].center * p.w, 0.0);
        float sphereDist = sphereSDF(spherePoint, spheres[0].radius);
        ClosestPoint sphereCP = ClosestPoint(sphereDist, spheres[0].mat);
        resCP = opUnion(resCP, sphereCP);
        }

        // Merged sphere and torus.
        {
        vec4 spherePoint = p - vec4(spheres[1].center * p.w, 0.0);
        float sphereDist = sphereSDF(spherePoint, spheres[1].radius);
        ClosestPoint sphereCP = ClosestPoint(sphereDist, spheres[1].mat);

        vec4 torusPoint = p - vec4((tori[0].translation + vec3(1.5, 0.0, -1.5) * sin(currentTime) * float(sceneNum == 2)) * p.w, 0.0);
        float torusDist = torusSDF(torusPoint, tori[0].params);
        ClosestPoint torusCP = ClosestPoint(torusDist, tori[0].mat);
        torusCP = opSmoothUnion(sphereCP, torusCP, 1.0);
        resCP = opUnion(resCP, torusCP);
        }

        // Subtract a sphere from a box.
        {
        vec3 displacement = vec3(0.0, 0.5, 0.0) * sin(3.0 * currentTime + 1.23);
        vec4 spherePoint = p - vec4((spheres[2].center + displacement * float(sceneNum == 2)) * p.w, 0.0);
        float sphereDist = sphereSDF(spherePoint, spheres[2].radius);
        ClosestPoint sphereCP = ClosestPoint(sphereDist, spheres[2].mat);

        vec4 boxPoint = p - vec4(boxes[0].translation * p.w, 0.0);
        boxPoint = opRotate(boxPoint, rotY(radians(45.0)));
        float boxDist = boxSDF(boxPoint, boxes[0].scale);
        ClosestPoint boxCP = ClosestPoint(boxDist, boxes[0].mat);
        boxCP = opSmoothSubtraction(boxCP, sphereCP, 0.1);
        resCP = opUnion(resCP, boxCP);
        }

        // Rotate a torus.
        {
        vec4 torusPoint = p - vec4(tori[1].translation * p.w, 0.0);
        if (sceneNum == 2)
        {
            torusPoint = opRotate(torusPoint, rotX(0.59 * currentTime));
            torusPoint = opRotate(torusPoint, rotZ(0.83 * currentTime));
            // torusPoint = opTwist(torusPoint * 0.05, 3.5 * sin(currentTime) * float(sceneNum == 2));
        }

        float torusDist = torusSDF(torusPoint, tori[1].params);
        ClosestPoint torusCP = ClosestPoint(torusDist, tori[1].mat);
        resCP = opSmoothUnion(resCP, torusCP, 0.5);
        }
    }

    if (sceneNum == 3 || sceneNum == 4)
    {
        // 2D repeated spheres with wave.
        {
        vec4 spherePoint = p - vec4(spheres[3].center * p.w, 0.0);
        if (sceneNum == 3)
        {
            spherePoint = opRepeat(spherePoint, vec3(3.0, 0.0, 3.0));
        }
        else
        {
            spherePoint = opRepeatWave(spherePoint, vec3(3.0, 0.0, 3.0));
        }
        float sphereDist = sphereSDF(spherePoint, spheres[3].radius);
        ClosestPoint sphereCP = ClosestPoint(sphereDist, spheres[3].mat);
        resCP = opUnion(resCP, sphereCP);
        }
    }

    if (sceneNum == 5)
    {
        // 3D repeated spheres.
        {
        vec4 spherePoint = p - vec4((spheres[4].center) * p.w, 0.0);
        spherePoint = opRepeat(spherePoint, vec3(12.0));
        float sphereDist = sphereSDF(spherePoint, spheres[4].radius);
        ClosestPoint sphereCP = ClosestPoint(sphereDist, spheres[4].mat);

        vec3 displacement = vec3(12.0 * sin(1.0 * currentTime), 0.0, 12.0 * cos(1.0 * currentTime));
        vec4 torusPoint = p - vec4((tori[0].translation + displacement) * p.w, 0.0);
        float torusDist = torusSDF(torusPoint, tori[0].params);
        ClosestPoint torusCP = ClosestPoint(torusDist, material_box);
        torusCP = opSmoothUnion(sphereCP, torusCP, 1.0);
        resCP = opUnion(resCP, torusCP);
        }
    }

    // A Sierpinski triangle.
    // TODO Fix this!!!
#if FALSE
    {
    // vec4 fractalPoint = opTranslate(p, vec3(0.0, 1.0, 0.0));
    vec4 fractalPoint = p;
    float fractalDist = sierpinskiTetrahedronSDF(fractalPoint, 1);
    ClosestPoint fractalCP = ClosestPoint(fractalDist, material_lambert);
    resCP = opUnion(resCP, fractalCP);
    }
#endif

    if (sceneNum == 6 || sceneNum == 7)
    {
        // A Menger sponge.
        {
        vec4 fractalPoint = opTranslate(p, vec3(0.0, 1.0, 0.0));
        if (sceneNum == 7)
        {
            fractalPoint = opRepeat(fractalPoint, vec3(5.0));
        }
        fractalPoint = opRotate(fractalPoint, rotY(currentTime));
        fractalPoint = opRotate(fractalPoint, rotX(currentTime));
        float fractalDist = mengerSpongeSDF(fractalPoint, 7);
        ClosestPoint fractalCP = ClosestPoint(fractalDist, material_lambert);
        resCP = opUnion(resCP, fractalCP);
        }
    }
    
    // A Mandelbulb.
    // TODO Fix this!!!
#if FALSE
    {
    vec4 fractalPoint = opTranslate(p, vec3(0.0, 1.0, 0.0));
    float fractalDist = mandelburbSDF(fractalPoint, 2);
    ClosestPoint fractalCP = ClosestPoint(fractalDist, material_lambert);
    resCP = opUnion(resCP, fractalCP);
    }
#endif

    if (sceneNum == 1 || sceneNum == 2 || sceneNum == 6)
    {
        // A simple plane.
        {
        float planeDist = planeSDF(p);
        ClosestPoint planeCP = ClosestPoint(planeDist, groundPlane.mat);
        resCP = opUnion(resCP, planeCP);
        }
    }
    return resCP;
}


// Procedural generation: Ocean surface.
float heightMap(vec2 coord)
{
    // DC term.
    const float WATER_LEVEL = -25.0;

    // Large variations.
    const vec4 WAVE_NUMS_X = vec4(0.021, 0.0124, 0.00117, 0.02358);
    const vec4 WAVE_NUMS_Y = vec4(0.00103, 0.00992, 0.00496, 0.0123);
    const vec4 ANG_VELS = vec4(1.0, 1.228, 0.0982, 4.2752);
    const vec4 WAVE_HEIGHTS = vec4(5.0, 4.0, -3.0, 6.0);

    // Small variations.
    const vec2 FBM_AMPLITUDE = vec2(-0.25, 0.59);
    const vec2 FBM_WAVE_NUMS = vec2(0.0039, 0.0427);
    const vec2 FBM_ANG_VALS = vec2(0.4997, 3.412);
    const vec2 FBM_HURST_EXPONENT = vec2(1.0, 0.49);
    const ivec2 FBM_NUM_OCTAVES = ivec2(3, 7);
    const mat3 FBM_TRANSFORM = mat3(
        vec3(1.6, -1.2, 0.0),
        vec3(1.2,  1.6, 0.0),
        vec3(0.0,  0.0, 2.037)
    );
    
    // Temporal variance.
    const vec2 ANG_VEL_MULTIPLIER_1 = vec2(0.101, 0.085);
    const vec2 ANG_VEL_MULTIPLIER_2 = vec2(0.187, -0.129);

    vec2 tCoord1 = currentTime * ANG_VEL_MULTIPLIER_1;
    vec2 tCoord2 = currentTime * ANG_VEL_MULTIPLIER_2;

    float height = WATER_LEVEL;
    for (int i = 0; i < 4; ++i)
    {
        height += WAVE_HEIGHTS[i] * sin(
            WAVE_NUMS_X[i] * coord.x + WAVE_NUMS_Y[i] * coord.y + ANG_VELS[i] * tCoord1.x
        );
    }
    height += FBM_AMPLITUDE[0] * fBM(
        FBM_WAVE_NUMS[0] * coord + FBM_ANG_VALS[0] * tCoord1,
        FBM_HURST_EXPONENT[0],
        FBM_NUM_OCTAVES[0]
    );
    height += FBM_AMPLITUDE[1] * fBM(
        vec3(FBM_WAVE_NUMS[1] * coord, FBM_ANG_VALS[1] * tCoord2),
        FBM_TRANSFORM,
        FBM_HURST_EXPONENT[1],
        FBM_NUM_OCTAVES[1]
    );
    return height;
}

float densityMap(vec3 coord)
{
    // Variations.
    const vec2 FBM_AMPLITUDE = vec2(1.0, 0.2);
    const vec2 FBM_SCALE = vec2(0.00081208, 0.0129392);
    const vec2 FBM_ANG_VALS = vec2(0.12, 0.37);
    const vec2 FBM_HURST_EXPONENT = vec2(0.4, 0.123);
    const ivec2 FBM_NUM_OCTAVES = ivec2(12, 4);

    // Temporal variance.
    const vec3 ANG_VEL_MULTIPLIER_1 = vec3(2.89, -0.003, 1.3);
    const vec3 ANG_VEL_MULTIPLIER_2 = vec3(5.87, -0.005, 3.23);

    vec3 tCoord1 = currentTime * ANG_VEL_MULTIPLIER_1;
    vec3 tCoord2 = currentTime * ANG_VEL_MULTIPLIER_2;
    
    if (coord.y < CLOUD_HEIGHT_MIN || coord.y > CLOUD_HEIGHT_MAX)
        return 0.0;
    
    float density = 0.0;
    density += FBM_AMPLITUDE[0] * fBM(
        FBM_SCALE[0] * coord + FBM_ANG_VALS[0] * tCoord1,
        FBM_HURST_EXPONENT[0],
        FBM_NUM_OCTAVES[0]
    );
#if FALSE
    density += FBM_AMPLITUDE[1] * fBM(
        FBM_SCALE[1] * coord + FBM_ANG_VALS[1] * tCoord2,
        FBM_HURST_EXPONENT[1],
        FBM_NUM_OCTAVES[1]
    );
#endif
    return density;
}

// For debugging density map.
float heightMapDensity(vec2 coord)
{
    float maxDensity = 0.0;
    float totDensity = 0.0;
    int j = 0;
    for (float i = CLOUD_HEIGHT_MIN; i <= CLOUD_HEIGHT_MAX; i += 2.0)
    {
        ++j;
        float density = densityMap(vec3(coord.x, i, coord.y));
        maxDensity = max(density, maxDensity);
        totDensity += density;
    }
    totDensity /= j;
    return totDensity;
}


// Ray marching algorithm.
Ray getRay(vec2 uv)
{
    Ray ray;
    ray.origin = cameraPosition;
    uv = 2.0 * uv - 1.0;
    uv.x *= (resolution.x / resolution.y);
    vec3 dir = vec3(uv, -2.0 / tan(fovY * 0.5));
    ray.direction = normalize(cam2worldRotMatrix * dir);
    return ray;
}

bool rayMarch(inout Ray ray, inout HitRecord hit, float sharpness)
{
    hit.s = MAX_MARCHING_STEPS;
    hit.t = 1.0 / 0.0; // Infinity
    hit.p = ray.origin;
    hit.normal = ray.direction;
    hit.vis = 1.0;

    float depth = 0.0;
    float lastDist = 1.0 / 0.0;
    for (int i = 0; i < MAX_MARCHING_STEPS; ++i)
    {
        vec3 currentPos = ray.origin + depth * ray.direction;
        ClosestPoint cp = sceneSDF(vec4(currentPos, 1.0));
        if (cp.dist < EPSILON)
        {
            hit.s = i;
            hit.t = depth;
            hit.p = currentPos;
            hit.normal = getNormal(vec4(currentPos, 1.0), EPSILON);
            hit.vis = 0.0;
            hit.mat = cp.mat;
            return true;
        }
#ifdef OPTION_DETAILED_SHADOW
        float distSq = cp.dist * cp.dist;
        float windowWidth = distSq * 0.5 / lastDist;
        float newVis = sharpness
                * sqrt(distSq - windowWidth * windowWidth)
                / max(depth - windowWidth, 0.0);
#else
        float newVis = sharpness * cp.dist / depth;
#endif
        hit.vis = min(hit.vis, newVis);
        lastDist = cp.dist;
        depth += cp.dist;
        if (depth > MAX_DIST)
        {
            return false;
        }
    }
    return false;
}

bool rayMarchHeightMap(inout Ray ray, inout HitRecord hit, float sharpness)
{
    const float STEP_SIZE = 1.0;
    const float STEP_SIZE_THRESHOLD = 300.0;
    const float STEP_SIZE_MULTIPLIER_SKY = 50.0;
    const float THRESHOLD = 1e-3;
    const float HEIGHTMAP_EPSILON = 1.0;
    const float HEIGHTMAP_SMOOTHNESS = 1.0;

    hit.s = MAX_MARCHING_STEPS;
    hit.t = 1.0 / 0.0; // Infinity
    hit.p = ray.origin;
    hit.normal = ray.direction;
    hit.vis = 1.0;

    float fog = 0.0;
    float depth = 0.0;
    for (int i = 0; i < MAX_MARCHING_STEPS; ++i)
    {
        // Determine the step size.
        float stepSize = STEP_SIZE;
        if (depth > STEP_SIZE_THRESHOLD)
        {
            float multiplier = depth / STEP_SIZE_THRESHOLD;
            multiplier = multiplier * multiplier * multiplier;
            stepSize *= multiplier;
        }

        // March the ray.
        vec3 currentPos = ray.origin + depth * ray.direction;
        float yDiff = currentPos.y - heightMap(currentPos.xz);
        if (yDiff < THRESHOLD)
        {
            hit.s = i;
            hit.t = depth;
            hit.p = currentPos - stepSize * ray.direction;
            hit.normal = heightMapNormal(
                currentPos.xz,
                HEIGHTMAP_EPSILON,
                HEIGHTMAP_SMOOTHNESS
            );
            hit.vis = 0.0;
            hit.mat = material_water;
            return true;
        }
        hit.vis = min(hit.vis, sharpness * stepSize / depth);

        if (ray.direction.y > 0.0)
        {
            stepSize *= STEP_SIZE_MULTIPLIER_SKY;
        }
        else
        {
            stepSize *= max(1.0, yDiff);
        }
        depth += stepSize;

        if (depth > MAX_DIST)
        {
            return false;
        }
    }
    return false;
}

bool rayMarchCloudMap(Ray ray, vec3 lightDir, inout HitRecord hit, out vec4 color)
{
    const float EXPLORATION_MULTIPLIER = 5.0;
    const int MAX_MARCHING_STEPS_PRIMARY = 30;
    const int MAX_MARCHING_STEPS_SECONDARY = 6;
    const float STEP_SIZE_PRIMARY = 7.0;
    const float STEP_SIZE_SECONDARY = 20.0;
    const float EXTINCTION_MULTIPLIER_PRIMARY = 0.019; // Determines the transparency.
    const float EXTINCTION_MULTIPLIER_SECONDARY = 0.05; // Determines the color.

    color = vec4(0.0);

    float fog = 0.0;
    float depth = (CLOUD_HEIGHT_MIN - ray.origin.y) / ray.direction.y;
    bool flagHit = false;
    float densityTotal = 0.0;
    for (int i = 0; i < MAX_MARCHING_STEPS_PRIMARY; ++i)
    {
        // Determine the step size.
        float stepSize = STEP_SIZE_PRIMARY;
        if (!flagHit)
        {
            stepSize *= EXPLORATION_MULTIPLIER;
        }

        // March the ray.
        vec3 currentPos = ray.origin + depth * ray.direction;
        float density = densityMap(currentPos);
        if (!flagHit && density > CLOUD_DENSITY_THRESHOLD)
        {
            flagHit = true;
            hit.s = i;
            hit.t = depth;
            hit.p = currentPos - stepSize * ray.direction;
        }
        if (currentPos.y > CLOUD_HEIGHT_MAX)
        {
            break;
        }

        // Generate the second ray to evaluate the light intensity per step.
        if (flagHit)
        {
            // Density is accumulated when the cloud is hit.
            densityTotal += density * stepSize;

            float densityTotal2 = 0.0;
            for (int j = 0; j < MAX_MARCHING_STEPS_SECONDARY; ++j)
            {
                float dist = STEP_SIZE_SECONDARY * j;
                vec3 currentPos2 = hit.p + dist * lightDir;

                float density2 = densityMap(currentPos2);
                densityTotal2 += density2 * STEP_SIZE_SECONDARY;
                if (currentPos2.y > CLOUD_HEIGHT_MAX)
                    break;
            }
            color.rgb += exp(
                -EXTINCTION_MULTIPLIER_SECONDARY * CLOUD_EXTINCTION
                * (densityTotal + densityTotal2)
            );
        }

        // stepSize *= min(1.0, 0.6 * ((1.0 - density) * (1.0 - density) + 1.0));
        depth += stepSize;
    }
    // Alpha channel is determined by the primary ray.
    color.a = clamp(max(
        1.0 - CLOUD_MAXIMUM_ATTENUATION,
        exp(-EXTINCTION_MULTIPLIER_PRIMARY * densityTotal)
    ), 0.0, 1.0);
    color.rgb = clamp(color.rgb, 0.0, 1.0);
    return flagHit;
}


// Illumination and scattering.
bool volumetricAttenuation(Ray ray, vec3 lightDir, vec3 lightColor, out vec4 color)
{
    HitRecord cloudHit;
    bool anyHit = rayMarchCloudMap(ray, lightDir, cloudHit, color);
    return anyHit;
}

vec3 hardShadow(HitRecord hit, vec3 lightDir)
{
    // Generate a shadow ray.
    Ray shadowRay = Ray(
        hit.p + hit.normal * SCATTERED_RAY_BIAS,
        lightDir
    );
    HitRecord shadowHit;
    bool anyHit = rayMarch(shadowRay, shadowHit, SHADOW_SHARPNESS);
    return vec3(float(!anyHit));
}

vec3 softShadow(HitRecord hit, vec3 lightDir)
{
    // Generate a shadow ray.
    Ray shadowRay = Ray(
        hit.p + hit.normal * SCATTERED_RAY_BIAS,
        lightDir
    );
    HitRecord shadowHit;
    rayMarch(shadowRay, shadowHit, SHADOW_SHARPNESS);
    return vec3(shadowHit.vis);
}

vec3 calculateDiffuseSpecular(HitRecord hit, vec3 lightDir, vec3 lightColor, bool castShadow)
{
    vec3 shadowAttn = vec3(1.0);
#ifdef OPTION_SHADOW_ON
    if (useShadow && castShadow)
#ifdef OPTION_SOFT_SHADOW
        shadowAttn = softShadow(hit, lightDir);
#else
        shadowAttn = hardShadow(hit, lightDir);
#endif
#endif

    // 2. Diffuse
    float diffuseCosine = max(dot(hit.normal, lightDir), 0.0);
    vec3 diffuse = diffuseCosine * hit.mat.Kd;

    // 3. Specular
    vec3 viewDir = normalize(cameraPosition - hit.p);
    vec3 reflectDir = reflect(-lightDir, hit.normal);
    float specularCosine = max(dot(viewDir, reflectDir), 0.0);
    float specularFactor = pow(specularCosine, hit.mat.shininess);
    vec3 specular = specularCosine * hit.mat.Ks;

    // Phong lighting for each light sources.
    return shadowAttn * (specular + diffuse) * lightColor;
}

vec3 phongIllumination(Ray ray, HitRecord hit)
{
    // Do Phong lighting.
    // 1. Ambient
    vec3 ambient = hit.mat.Ka;
    vec3 phong = ambient * ambientLightColor;

    // 2. Diffuse and specular.

    // Directional light source.
    phong += calculateDiffuseSpecular(
        hit, -sun.direction, sun.color, sun.castShadow
    );

    // Point light sources.
    for (int i = 0; i < NUM_POINT_LIGHTS; ++i)
    {
        // Calculate shadow with additional ray casting.
        vec3 lightDir = normalize(pointlights[i].position - hit.p);

        // Phong lighting for each light sources.
        phong += calculateDiffuseSpecular(
            hit, lightDir, pointlights[i].color, pointlights[i].castShadow
        );
    }

    // Area light sources.
    // Assume all area lights to be triangular for brevity.
    // for (int i = 0; i < NUM_AREA_LIGHTS; ++i)
    // {
    //     // Calculate shadow with additional ray casting.
    //     // Light direction of an area light should be arbitraty.
    //     vec3 lightPos = randTriangle(hit.p + i, arealights[i].geom);
    //     vec3 lightDir = normalize(lightPos - hit.p);
    //
    //     // Phong lighting for each light sources.
    //     phong += calculateDiffuseSpecular(
    //         hit, lightDir, arealights[i].color, arealights[i].castShadow
    //     );
    // }

    return clamp(phong, 0.0, 1.0);
}


// Scattering of the rays.
bool mirrorScatter(HitRecord hit, inout Ray ray, inout vec3 attenuation)
{
    float cosine = dot(ray.direction, hit.normal);
    vec3 rayBiasedOrigin = hit.p + -sign(cosine) * hit.normal * SCATTERED_RAY_BIAS;
    vec3 reflection = reflect(ray.direction, hit.normal);
    ray = Ray(rayBiasedOrigin, normalize(reflection));
    attenuation *= schlick(abs(cosine), hit.mat.R0);
    return true;
}

bool lambertianScatter(HitRecord hit, inout Ray ray, inout vec3 attenuation)
{
    float cosine = dot(ray.direction, hit.normal);
    vec3 rayBiasedOrigin = hit.p + -sign(cosine) * hit.normal * SCATTERED_RAY_BIAS;
    ray = Ray(rayBiasedOrigin, normalize(hit.normal + randUnitSphere(hit.p)));
    attenuation *= hit.mat.R0;
    return true;
}

bool refractBool(vec3 v, vec3 n, float eta, out vec3 refracted)
{
    float cosi = dot(v, n);
    float costsq = 1.0 - eta * eta * (1 - cosi * cosi);
    if (costsq > 0)
    {
        refracted = eta * (v - n * cosi) - n * sqrt(costsq);
        return true;
    }
    else
        return false;
}

bool refractiveScatter(HitRecord hit, inout Ray ray, inout vec3 attenuation)
{
    // Calculate the refraction/reflection ratio and determine the next ray.
    float eta = 1.0 / hit.mat.ior; // ni/nt
    float cosine = -dot(ray.direction, hit.normal);
    float dir = sign(cosine);
    // Ray is inside the material.
    if (dir < 0.0)
    {
        eta = hit.mat.ior;
        cosine *= -eta;

        // Apply Beer-Lambert law to importance sample the next ray.
        vec3 kappa = hit.mat.extinction;
        vec3 transmittance = exp(-kappa * hit.t);
        attenuation *= transmittance;
    }

    // Reflect or refract according to the reflection ratio.
    float rv1 = hash1D(ray.direction.xy + hit.p.zy);

    float reflectRatio = float(1.0);
    vec3 refraction = vec3(0.0);
    if (refractBool(ray.direction, dir * hit.normal, eta, refraction))
        reflectRatio = schlick(cosine, eta);

    vec3 reflection = reflect(ray.direction, hit.normal);
    vec3 rayBiasedOrigin = hit.p - dir * hit.normal + SCATTERED_RAY_BIAS;
    if (rv1 < reflectRatio)
        ray = Ray(rayBiasedOrigin, normalize(reflection));
    else
        ray = Ray(rayBiasedOrigin, normalize(refraction));
    return true;
}

bool specularScatter(HitRecord hit, inout Ray ray, inout vec3 attenuation)
{
    float cosine = dot(ray.direction, hit.normal);
    vec3 rayBiasedOrigin = hit.p + -sign(cosine) * hit.normal * SCATTERED_RAY_BIAS;
    vec3 reflection = reflect(ray.direction, hit.normal);
    ray = Ray(
        rayBiasedOrigin,
        normalize(reflection + 0.0001 * randUnitSphere(hit.p))
    );
    attenuation *= schlick(abs(cosine), hit.mat.R0);
    return dot(ray.direction, hit.normal) > 0.0;
}



void main()
{
    // Return the default (miss) color.
    vec3 color = vec3(0.0);

    // Do anti-aliasing.
    for (int s = 0; s < ANTIALIASING_SAMPLES; ++s)
    {
        bool flagStopIteration = false;
        vec2 coord = TexCoord + ANTIALIASING_JITTER * randUnitDisk(vec3(TexCoord.yx, float(s)));

        Ray ray = getRay(coord);
        HitRecord hit;
        vec3 attenuation = vec3(1.0);
        for (int b = 0; b < MAX_DEPTH; ++b)
        {
            float sunCosine = max(0.0, dot(ray.direction, -sun.direction));

            bool anyHit = rayMarch(ray, hit, SHADOW_SHARPNESS);
            if (!anyHit)
            {
                bool heightMapHit = false;
#ifdef OPTION_RENDER_WATER
                if (sceneNum == 8 || sceneNum == 9)
                {
                    heightMapHit = rayMarchHeightMap(ray, hit, SHADOW_SHARPNESS);
                }
#endif
                if (!heightMapHit)
                {
                    // Calculate the color of the skybox.
                    vec3 skyColor = texture(envCubeMap, ray.direction).rgb;

#ifdef OPTION_RENDER_SUN
                    // Calculate the color of the solar disc.
                    if (renderSun)
                    {
                        float sunStrength = pow(sunCosine, SUN_SHARPNESS);
                        skyColor *= (1.0 - sunStrength);
                        vec3 sunColor = clamp(sun.color * (
                            SUN_BRIGHTNESS * SUN_COLOR_MULTIPLIER * sunStrength
                            + SUN_CORONA_BRIGHTNESS * SUN_CORONA_COLOR_MULTIPLIER
                                * pow(sunCosine, SUN_CORONA_SHARPNESS)
                        ), 0.0, 1.0);
                        skyColor += sunColor;
                    }
#endif

#ifdef OPTION_RENDER_CLOUDS
                    // Calculate the color of the cloud if hit any.
                    if (sceneNum == 9)
                    {
                        vec4 cloudColor = vec4(0.0);
                        bool cloudHit = volumetricAttenuation(ray, sun.direction, sun.color, cloudColor);
                        if (cloudHit)
                        {
                            skyColor = mix(cloudColor.rgb, skyColor, cloudColor.a);
                            // skyColor = cloudColor.rgb;
                            // skyColor = cloudColor.aaa;
                            // deltaColor = vec3(0.0);
                        }
                    }
#endif

                    color += attenuation * skyColor;
                    break;
#ifdef OPTION_RENDER_WATER
                }
#endif
            }

#ifdef OPTION_RENDER_NORMAL
            color += hit.normal;
            break;
#endif

#ifdef OPTION_RENDER_GLOW
            if (renderGlow || (sceneNum == 7))
            {
                color += GLOW_INTENSITY * vec3(float(hit.s)) / MAX_MARCHING_STEPS;
                break;
            }
#endif

            vec3 deltaColor = attenuation * phongIllumination(ray, hit);

#ifdef OPTION_FOG_ON
            deltaColor = applyFog(deltaColor, FOG_BASE_ATTENUATION, hit.t, sunCosine, attenuation);
#endif

            switch (hit.mat.scatterType)
            {
            case SCATTER_TYPE_MIRROR:
                color += deltaColor;
                flagStopIteration = !mirrorScatter(hit, ray, attenuation);
                break;
            case SCATTER_TYPE_LAMBERTIAN:
                color += deltaColor;
                flagStopIteration = !lambertianScatter(hit, ray, attenuation);
                break;
            case SCATTER_TYPE_REFRACTIVE:
                flagStopIteration = !refractiveScatter(hit, ray, attenuation);
                break;
            case SCATTER_TYPE_SPECULAR:
                color += deltaColor;
                flagStopIteration = !specularScatter(hit, ray, attenuation);
                break;
            default:
                // No illumination.
                flagStopIteration = true;
                break;
            }

            // Next iteration.
            if (flagStopIteration)
                break;
        }
    }
    color /= ANTIALIASING_SAMPLES;

	FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}

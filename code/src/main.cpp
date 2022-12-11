#define GLM_ENABLE_EXPERIMENTAL

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>

#include "base/system_global.h"
#include "base/engine_global.h"

#include "base/camera.h"
#include "base/scene.h"

#include "data/geometry.h"
#include "data/light.h"
#include "data/mesh.h"
#include "data/model.h"
#include "data/shader.h"
#include "data/texture.h"
#include "data/texture_cube.h"

#include "utils/math_utils.h"
#include "utils/screen_utils.h"

using namespace std::string_literals;


// Callbacks and interfaces
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);


int main()
{
    // GLFW: Initialize and configure.
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW"s << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    // Uncomment this statement to fix compilation on OS X.
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // GLFW Window creation.
    GLFWwindow* window = glfwCreateWindow(
        screen.width,
        screen.height,
        screen.name.c_str(),
        nullptr,
        nullptr
    );
    if (window == nullptr)
    {
        std::cout << "Failed to create GLFW window"s << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Tell GLFW to capture our mouse.
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // GLAD: Load all OpenGL function pointers.
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD"s << std::endl;
        return -1;
    }

    // Configure global OpenGL state.
    // glEnable(GL_DEPTH_TEST);

    // Create scenes and declare assets.
    // TODO replace this whole initialization to parsing some serialized database.
    engine::Scene* scene = new engine::Scene();

    // Build and compile our shader program.
    engine::Shader* rmShader = new engine::Shader(
        "Ray-Marching Shader"s,
        "../shaders/shader_ray_march.vert"s,
        "../shaders/shader_ray_march.frag"s
    );
    scene->addShader(rmShader);


    // Rendering screen.
    engine::Geometry* quadGeometry = new engine::Geometry(
        "Quad"s, "../resources/shape_primitive/quadPT.json"s
    );
    scene->addGeometry(quadGeometry);

    engine::CubemapTexture* skyboxTexture = new engine::CubemapTexture(
        "Default Skybox"s, 
        std::vector<std::string> {
            "../resources/cubemap/siurana/px.jpg"s,
            "../resources/cubemap/siurana/nx.jpg"s,
            "../resources/cubemap/siurana/py.jpg"s,
            "../resources/cubemap/siurana/ny.jpg"s,
            "../resources/cubemap/siurana/pz.jpg"s,
            "../resources/cubemap/siurana/nz.jpg"s
        }
    );
    scene->addCubemapTexture(skyboxTexture);


    // TODO : Maintain a light array in the scene.
    engine::DirectionalLight* sun = new engine::DirectionalLight(
        "Sun"s, 30.0f, 30.0f, glm::vec3(0.9f)
    );
    // Install our sun.
    gameManager.sun = sun;


    // unsigned int rmShaderUBI = glGetUniformBlockIndex(rmShader->ID, "mesh_vertices_ubo");
    // glUniformBlockBinding(rmShader->ID, rmShaderUBI, 0);

    // Use this code for mesh render
    // unsigned int uboMeshVertices;
    // glGenBuffers(1, &uboMeshVertices);
    // glBindBuffer(GL_UNIFORM_BUFFER, uboMeshVertices);
    // glBufferData(GL_UNIFORM_BUFFER, 4000 * sizeof(float), &icosphereModel->mesh->vertices[0], GL_STATIC_DRAW);
    // glBindBufferRange(GL_UNIFORM_BUFFER, 0, uboMeshVertices, 0, 4000 * sizeof(float));
    // glBindBuffer(GL_UNIFORM_BUFFER, 0);


    // Presets.
    rmShader->use();
    rmShader->setFloat("envCubeMap"s, 0.0f);

    // Render loop.
    while (!glfwWindowShouldClose(window))
    {
        // Get current camera.
		engine::Camera* currentCamera = gameManager.defaultCamera;

        // Maintain loop time.
        timer.update((float)glfwGetTime());

        // Process input.
        processInput(window);
        gameManager.processCommands(&cmd);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        rmShader->use();
        rmShader->setFloat("currentTime"s, timer.getLastFrame());
        rmShader->setVec2("resolution"s, glm::vec2(
            (GLfloat)screen.width,
            (GLfloat)screen.height
        ));
        rmShader->setFloat("fovY"s, glm::radians(currentCamera->zoom));
        rmShader->setVec3("cameraPosition"s, currentCamera->tf.position);
        rmShader->setMat3(
            "cam2worldRotMatrix"s,
            glm::transpose(glm::mat3(currentCamera->getViewMatrix()))
        );

        rmShader->setVec3("ambientLightColor"s, sun->lightColor * 0.02f);
        rmShader->setVec3("sun.direction"s, sun->lightDir);
        rmShader->setVec3("sun.color"s, sun->lightColor);
        rmShader->setBool("sun.castShadow"s, true);

        rmShader->setInt("sceneNum"s, graphicsSettings.demoMode);
        rmShader->setBool("renderSun"s, graphicsSettings.renderSun);
        rmShader->setBool("useShadow"s, graphicsSettings.useShadow);
        rmShader->setBool("renderGlow"s, graphicsSettings.renderGlow);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture->ID);

        glBindVertexArray(quadGeometry->VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        std::cout << "useShadow  " << graphicsSettings.useShadow << "  renderGlow  " << graphicsSettings.renderGlow << "   ";

        std::cout << std::setprecision(3) << "pos( "
            << gameManager.defaultCamera->tf.position.x << ", "
            << gameManager.defaultCamera->tf.position.y << ", "
            << gameManager.defaultCamera->tf.position.z << " );  rot( "
            << gameManager.defaultCamera->tf.rotation.x << ", "
            << gameManager.defaultCamera->tf.rotation.y << ", "
            << gameManager.defaultCamera->tf.rotation.z << " );  sun( "
            << gameManager.sun->lightDir.x << ", "
            << gameManager.sun->lightDir.y << ", "
            << gameManager.sun->lightDir.z << " )" << std::endl;

        // GLFW: Swap buffers and poll IO events (keys pressed/released, mouse moved etc.).
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Deallocate the loaded assets.
    delete scene;

    // GLFW: Terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    return 0;
}

void setToggle(GLFWwindow* window, unsigned int key, bool *value)
{
    if (glfwGetKey(window, key) == GLFW_PRESS && !screen.isKeyboardDone[key])
    {
        *value = !*value;
        screen.isKeyboardDone[key] = true;
    }
    if (glfwGetKey(window, key) == GLFW_RELEASE)
    {
        screen.isKeyboardDone[key] = false;
    }
}

// Process all input: query GLFW whether relevant keys are pressed/released
// this frame and react accordingly.
void processInput(GLFWwindow* window)
{
    static float toggleShadowLastFrame = 0.0f;
    static float toggleSunLastFrame = 0.0f;
    static float toggleGlowLastFrame = 0.0f;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Make camera move by 3D translation (WASD, arrows, ascention using space).
    int moveForward = 0;
    int moveRight = 0;
    int moveUp = 0;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        ++moveForward;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        --moveForward;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        --moveRight;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        ++moveRight;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        ++moveUp;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        --moveUp;
    }
    
    // Sprint with shift key.
    int sprint = 0;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        ++sprint;
    }

    // Arrow key : increase, decrease sun's azimuth, elevation with amount of t.
    int azimuthEast = 0;
    int elevationUp = 0;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        ++elevationUp;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        --elevationUp;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        --azimuthEast;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        ++azimuthEast;
    }

    // Demo mode to show.
    int demoMode = graphicsSettings.demoMode;
    if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS)
    {
        demoMode = 0;
    }
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        demoMode = 1;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        demoMode = 2;
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
    {
        demoMode = 3;
    }
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
    {
        demoMode = 4;
    }
    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS)
    {
        demoMode = 5;
    }
    if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS)
    {
        demoMode = 6;
    }
    if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS)
    {
        demoMode = 7;
    }
    if (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS)
    {
        demoMode = 8;
    }
    if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS)
    {
        demoMode = 9;
    }

    float lastFrame = timer.getLastFrame();
    bool toggleShadow = false;
    bool toggleSun = false;
    bool toggleGlow = false;
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
    {
        if (toggleShadowLastFrame + engine::KEYBOARD_TOGGLE_DELAY < lastFrame)
        {
            toggleShadow = true;
            toggleShadowLastFrame = lastFrame;
        }
    }
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
    {
        if (toggleSunLastFrame + engine::KEYBOARD_TOGGLE_DELAY < lastFrame)
        {
            toggleSun = true;
            toggleSunLastFrame = lastFrame;
        }
    }
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
    {
        if (toggleGlowLastFrame + engine::KEYBOARD_TOGGLE_DELAY < lastFrame)
        {
            toggleGlow = true;
            toggleGlowLastFrame = lastFrame;
        }
    }

    // Log current position and rotation.
    // if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
    // {
    //     std::cout << std::setprecision(3) << "pos( "
    //         << gameManager.defaultCamera->tf.position.x << ", "
    //         << gameManager.defaultCamera->tf.position.y << ", "
    //         << gameManager.defaultCamera->tf.position.z << " );  rot( "
    //         << gameManager.defaultCamera->tf.rotation.x << ", "
    //         << gameManager.defaultCamera->tf.rotation.y << ", "
    //         << gameManager.defaultCamera->tf.rotation.z << " )" << std::endl;
    // }

    // Take a screenshot.
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS && screen.isKeyboardDone[GLFW_KEY_V] == false)
    {
        time_t t = time(NULL);
        struct tm tm = *localtime(&t);
        char date_char[128];
        sprintf(
            date_char, "%d_%d_%d_%d_%d_%d.png",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec
        );
        saveImage(date_char);
        screen.isKeyboardDone[GLFW_KEY_V] = true;
    }
    else if (glfwGetKey(window, GLFW_KEY_V) == GLFW_RELEASE)
    {
        screen.isKeyboardDone[GLFW_KEY_V] = false;
    }

    // Toggle fullscreen ? TODO
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS && screen.isKeyboardDone[GLFW_KEY_Z] == false)
    {
        const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
        if (screen.isWindowed)
        {
            glfwSetWindowMonitor(
                window, glfwGetPrimaryMonitor(), 0, 0,
                mode->width, mode->height, mode->refreshRate
            );
        }
        else
        {
            glfwSetWindowMonitor(
                window, nullptr, 0, 0,
                screen.width, screen.height, mode->refreshRate
            );
        }
        screen.isWindowed = !screen.isWindowed;
        screen.isKeyboardDone[GLFW_KEY_Z] = true;
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_RELEASE)
    {
        screen.isKeyboardDone[GLFW_KEY_Z] = false;
    }

    // Update the commands.
    cmd.moveForward = moveForward;
    cmd.moveRight = moveRight;
    cmd.moveUp = moveUp;
    cmd.sprint = sprint;

    cmd.azimuthEast = azimuthEast;
    cmd.elevationUp = elevationUp;

    graphicsSettings.demoMode = demoMode;
    graphicsSettings.useShadow = toggleShadow
        ? !graphicsSettings.useShadow : graphicsSettings.useShadow;
    graphicsSettings.renderSun = toggleSun
        ? !graphicsSettings.renderSun : graphicsSettings.renderSun;
    graphicsSettings.renderGlow = toggleGlow
        ? !graphicsSettings.renderGlow : graphicsSettings.renderGlow;
}

// GLFW: Whenever the window size changed (by OS or user resize) this callback function executes.
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screen.width = width;
    screen.height = height;

    // Make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// GLFW: Whenever the mouse moves, this callback is called.
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (cmd.firstMouse)
    {
        cmd.lastX = (float)xpos;
        cmd.lastY = (float)ypos;
        cmd.firstMouse = false;
    }

    // Calculate how much cursor have moved, rotate camera proportional to the
    // value, using processMouseMovement.
    float xoffset = (float)xpos - cmd.lastX;
    float yoffset = cmd.lastY - (float)ypos;
    cmd.lastX = (float)xpos;
    cmd.lastY = (float)ypos;
    gameManager.processMouseMovement(xoffset, yoffset);
}

// GLFW: Whenever the mouse scroll wheel scrolls, this callback is called.
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    gameManager.processMouseScroll(xoffset, yoffset);
}

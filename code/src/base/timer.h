#ifndef TIMER_H
#define TIMER_H


namespace engine
{
class Timer
{
public:
	Timer() {}

	void update(float currentFrame)
	{
		deltaTime = currentFrame - this->lastFrame;
		this->lastFrame = currentFrame;
	}

	float getDeltaTime() { return this->deltaTime; }

    float getLastFrame() { return this->lastFrame; }

private:
	float deltaTime = 0.0f;
	float lastFrame = 0.0f;
};
}

#endif

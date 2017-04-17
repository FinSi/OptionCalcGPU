#include "stdafx.h"
#include "GPUThread.h"
#include <zthread/Thread.h>
#include <zthread/Semaphore.h>
#include <zthread/Mutex.h>

using namespace ZThread;

class CThreadTask : public Runnable {
public:

	CThreadTask(CUT_THREADROUTINE threadRoutine, 	void*	data):
		m_threadRoutine(threadRoutine),
		m_data(data)
	{}
	~CThreadTask(){}
	void run() {
		m_threadRoutine(m_data);
	}

private:
	CUT_THREADROUTINE m_threadRoutine;
	void*							m_data;
};

// Create thread.
GPUThread cutStartThread(CUT_THREADROUTINE threadRoutine, void * data)
{
	CThreadTask* threadTask = new CThreadTask(threadRoutine, data);
	Thread* thread = new Thread(threadTask);
	return thread;
}


bool cutEndThread(GPUThread t, unsigned long timeout)
{
	Thread* thread = static_cast<Thread*>(t);
	return thread->wait(timeout);
}

// Destroy thread.
void cutDestroyThread(GPUThread t)
{
	Thread* thread = static_cast<Thread*>(t);
	delete thread;
}

//create event
GPUEvent cutCreateEvent()
{
	Semaphore* sm = new Semaphore(0, 1);
	return sm;
}
//set event
void cutSetEvent(GPUEvent e)
{
	Semaphore* sm = static_cast<Semaphore*>(e);
	sm->post();
}
//wait event
bool cutTryWaitEvent(GPUEvent e, unsigned long timeout)
{
	Semaphore* sm = static_cast<Semaphore*>(e);
	return sm->tryWait(timeout);
}

void cutWaitEvent(GPUEvent e)
{
	Semaphore* sm = static_cast<Semaphore*>(e);
	return sm->wait();
}

//close event
void cutDestroyEvent(GPUEvent e)
{
	Semaphore* sm = static_cast<Semaphore*>(e);
	delete sm;
}



GPUMutex cutCreateMutex()
{
	Mutex* m = new Mutex();
	return m;
}

void cutLockMutex(GPUMutex m)
{
	Mutex* mutex = static_cast<Mutex*>(m);
	mutex->acquire();

}

bool cutTryLockMutex(GPUMutex m, unsigned long timeout)
{
	Mutex* mutex = static_cast<Mutex*>(m);
	return mutex->tryAcquire(timeout);
}

void cutUnlockMutex(GPUMutex m)
{
	Mutex* mutex = static_cast<Mutex*>(m);
	mutex->release();

}

void cutDestroyMutex(GPUMutex m)
{
	Mutex* mutex = static_cast<Mutex*>(m);
	delete mutex;
}

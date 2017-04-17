#ifndef __GPU_THREAD_H__
#define __GPU_THREAD_H__


typedef void *GPUThread;
typedef void *GPUEvent;
typedef void *GPUMutex;

#ifdef __cplusplus
extern "C" {
#endif

	///////////////////////
	//thread primitives
	///////////////////////
	typedef void (* CUT_THREADROUTINE)(void *);

	// Create thread.
	GPUThread cutStartThread(CUT_THREADROUTINE, void * data);

	// Wait for thread to finish.

	bool cutEndThread(GPUThread thread, unsigned long timeout);

	// Destroy thread.
	void cutDestroyThread(GPUThread thread);

	///////////////////////
	//event primitives
	///////////////////////
	
	//create event
	GPUEvent cutCreateEvent();
	//set event
	void cutSetEvent(GPUEvent e);
	//wait event
	bool cutTryWaitEvent(GPUEvent e, unsigned long timeout);
	void cutWaitEvent(GPUEvent e);
	//close event
	void cutDestroyEvent(GPUEvent e);

	///////////////////////
	//event primitives
	///////////////////////
	GPUMutex cutCreateMutex();

	void cutLockMutex(GPUMutex m);

	bool cutTryLockMutex(GPUMutex m, unsigned long timeout);

	void cutUnlockMutex(GPUMutex m);

	void cutDestroyMutex(GPUMutex m);


#ifdef __cplusplus
} // extern "C"
#endif



#endif // __GPU_THREAD_H__

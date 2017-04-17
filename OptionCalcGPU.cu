/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/


// includes, system
#include <math.h>

// includes, project
#include <cutil.h>
#define MAX_OPTION_COUNT 300000
#define STEPS_LENGTH 101

// includes, kernels
#include <OptionCalcGPU_kernel.cu>
#include <OptionCalcGPU_greeks.cu>
//#include <OptionCalcGPU_greeks.cu>
#include "OptionCalcGPU_sys.h"
#include "common.h"

#include "OptionCalcGPU_sys.h"
#include "GPUCommon.h"
#include "GPUThread.h"
#include "Logger.h"


GPU_DEVICE* g_pGPUList = NULL;
int g_nDeviceCount			= 0;
long g_isGPUInitialized = 0;

static long MAX_GPU_WAIT_TIMEOUT = 300000;

__host__ void createDataArrays(GPU_DEVICE* pGPU, long arrayLen);
__host__ void deleteDataArrays(GPU_DEVICE* pGPU);
__host__ void copyOptionBaseData(GPU_DATA*	pGPU,
																 float*			dSpotPrice,
																 float*			dStrike,
																 float*			dOptionPrice,
																 int*				nDTE,
																 int*				nIsCall,
																 int*				nIsAmerican,
																 long				arrayLen);

__host__ void copyRateData(	GPU_DATA*		pGPU,
														HOST_DATA*	pHD,
														float*			dDomesticRate,	
														float*			dForeignRate, 
														long				arrayLen,
														float				deltaRate = 0.0f);

__host__ void copyDividendData(		GPU_DATA*		pGPU,
																	HOST_DATA*	pHD,
																	float*		dSpotPrice, 
																	float*		divAmounts, 
																	float*		divYears, 
																	int*			divCount, 
																	int*			divIndex, 
																	long			divArrayLen,
																	long			Steps,
																	long			arrayLen);

__host__ void CachePayments(HOST_DATA*	pHD,
														float*			stockPrice, 
														float*			dYte, 
														float*			dContRd, 
														float*			DA, 
														float*			DT, 
														int*				DI, 
														int*				DC, 
														long				Steps, 
														long				optCount, 
														float*			dDiscount, 
														float*			Payment); 

__host__ float DividendsDiscount(float S, 
																 float* Payment, 
																 float T, 
																 float Rd, 
																 float* DA, 
																 float* DT, 
																 long DC, 
																 long Steps);

__host__ void dumpGPUArray(const char* name,
														float* p,
														long arrayLen);

__host__ void dumpGPUArray(const char* name,
														int* p,
														long arrayLen);

__host__ void dumpNativeArray(const char* name,
														double* p,
														long arrayLen);

__host__ void dumpNativeArray(const char* name,
														float* p,
														long arrayLen);

__host__ void dumpNativeArray(const char* name,
														int* p,
														long arrayLen);


__host__ void dumpAllGPUArrays(GPU_DATA*		pGPU,
															 HOST_DATA*		pHD,
															 const char*	name, 
															 long optArrayLen);



/*
__host__ float DividendPv(float dDivAmnt, 
													float dCntRate, 
													float dTime);

__host__ bool ValueNEQZero(float dVal)
{
	return fabs(dVal) > FLT_EPSILON;//DBL_EPSILON;
}
*/

__host__ void optionsPrepareAndEliminate(GPU_DATA*	pGPU,
																				 HOST_DATA*	pHD,
																				 float*			hVolatility, 
																				 float*			hVolatility1, 
																				 long				arrayLen, 
																				 long				Steps);

__host__ void calcOpt(GPU_DATA*	pGPU,
											long			arrayLen, 
											int				Steps);

__host__ void calcGreeks(GPU_DATA*	pGPU,
												 HOST_DATA*	pHD,
											long			arrayLen, 
											int				Steps);

__host__ void calcRhoGreek(	GPU_DATA*		pGPU,
														HOST_DATA*	pHD,
														long				arrayLen, 
														int					Steps);

__host__ void brentStep(double*			a, 
												double*			b, 
												double*			c,
												double*			d,
												double*			e,
												double*			fa,
												double*			fb,
												double*			fc,
												double			x1,  //float x1  = 0.01;
												double			x2,	//float x2  = 2.5;
												int*				calcFlag,
												int					iter,
												int					optCount,
												float*			V,
												float*			V1,
												double*			rV,
												float*			optionPrice);


long GPUInitImpl(GPU_DEVICE* pGPU)
{

  CUDA_SAFE_CALL(cudaSetDevice(pGPU->deviceID));

	CUT_DEVICE_INIT();

	WriteToLog("createDataArrays %d\n", MAX_OPTION_COUNT);

	createDataArrays(pGPU,MAX_OPTION_COUNT);

	dumpAllGPUArrays(&pGPU->GPUData, &pGPU->HOSTData, "createDataArrays", 100);

	return GPU_OK;
}

/*__declspec(dllexport)*/ long GPUInit()
{
	if (g_isGPUInitialized){
		return GPU_OK;
	}

	WriteToLog("GPUInit\n");

	cudaGetDeviceCount(&g_nDeviceCount);

	WriteToLog("cudaGetDeviceCount result %d\n", g_nDeviceCount);

	if (g_nDeviceCount < 1){
		return GPU_NOT_INITIALIZED;				
	}

//GPU_DEVICE

	g_pGPUList = (GPU_DEVICE*)malloc(g_nDeviceCount*sizeof(GPU_DEVICE));
	memset(g_pGPUList, '\0', g_nDeviceCount*sizeof(GPU_DEVICE));

	for (int i = 0; i < g_nDeviceCount; ++i)
	{	

		g_pGPUList[i].hGPUOperationEvent						= cutCreateEvent();
		g_pGPUList[i].hGPUOperationDoneEvent				= cutCreateEvent();

		g_pGPUList[i].deviceID											= i;

		//initialize critical section
		g_pGPUList[i].mutex	=	cutCreateMutex();
		cutLockMutex(g_pGPUList[i].mutex);
		
		g_pGPUList[i].gpuOperationType	= enGPUInit;
		g_pGPUList[i].hGPUWorkingThread = cutStartThread(GPUWorkingThread, &g_pGPUList[i]);

		cutSetEvent(g_pGPUList[i].hGPUOperationEvent);
		if (!cutTryWaitEvent(g_pGPUList[i].hGPUOperationDoneEvent, MAX_GPU_WAIT_TIMEOUT)){
			WriteToLog("Wait timeout in GPUInit. Device ID %d\n", i);
			cutUnlockMutex(g_pGPUList[i].mutex);
			return GPU_TIMEOUT;
		}

		cutUnlockMutex(g_pGPUList[i].mutex);

		if (g_pGPUList[i].curOperationStatus != GPU_OK){
			return g_pGPUList[i].curOperationStatus;
		}
	}

	g_isGPUInitialized = 1;

	return GPU_OK;
}

long GPUCloseImpl(GPU_DEVICE* pGPU)
{
	deleteDataArrays(pGPU);

	WriteToLog("deleteDataArrays\n");

	
	g_isGPUInitialized = 0;
	return GPU_OK;
}

/*__declspec(dllexport)*/ long GPUClose()
{
	WriteToLog("GPUClose\n");

	if (!g_isGPUInitialized){
		return GPU_OK;
	}

	for (int i = 0; i < g_nDeviceCount; ++i)
	{	

		cutLockMutex(g_pGPUList[i].mutex);		


		g_pGPUList[i].gpuOperationType	= enGPUClose;

		cutSetEvent(g_pGPUList[i].hGPUOperationEvent);
		if (!cutTryWaitEvent(g_pGPUList[i].hGPUOperationDoneEvent, MAX_GPU_WAIT_TIMEOUT)){
			WriteToLog("Wait timeout in GPUClose. Device ID %d\n", i);
		}

		if (!cutEndThread(g_pGPUList[i].hGPUWorkingThread, MAX_GPU_WAIT_TIMEOUT)){
			WriteToLog("Wait timeout in GPUClose. Device ID %d\n", i);
		}
		cutDestroyThread(g_pGPUList[i].hGPUWorkingThread);

		cutDestroyEvent(g_pGPUList[i].hGPUOperationEvent);
		cutDestroyEvent(g_pGPUList[i].hGPUOperationDoneEvent);

		cutDestroyMutex(g_pGPUList[i].mutex);

	}

	g_nDeviceCount		= 0;
	g_isGPUInitialized	= 0;
	free(g_pGPUList);

	return GPU_OK;
}

long GPUCalcVolatilityImpl(GPU_DEVICE*	pGPU,
													 float*				fDomesticRate,	
													 float*				fForeignRate, 
													 float*				fSpotPrice,
													 float*				fOptionPrice,
													 float*				fStrike,
													 int*					nDTE,
													 int*					nIsCall,
													 int*					nIsAmerican,
													 int*					divIndex,
													 int*					divCount,
													 float*				divAmounts,
													 float*				divYears,
													 long					divArrayLen,
													 long					optArrayLen,
													 long					nSteps,
													 float*				fVolatility)
{
	dumpNativeArray("DomesticRate", fDomesticRate, 100);
	dumpNativeArray("ForeignRate", fForeignRate, 100);
	dumpNativeArray("SpotPrice", fSpotPrice, 100);
	dumpNativeArray("OptionPrice", fOptionPrice, 100);
	dumpNativeArray("Strike", fStrike, 100);
	dumpNativeArray("DTE", nDTE, 100);
	dumpNativeArray("IsCall", nIsCall, 100);
	dumpNativeArray("IsAmerican", nIsAmerican, 100);

	HOST_DATA*	pHD = &pGPU->HOSTData;
	GPU_DATA*		pGD = &pGPU->GPUData;

	dumpAllGPUArrays(pGD, pHD, "copyOptionBaseData", min((int)optArrayLen, 100));

	WriteToLog("EnterCriticalSection completed\n");

	copyOptionBaseData(
		pGD,
		fSpotPrice,
		fStrike,
		fOptionPrice,
		nDTE,
		nIsCall,
		nIsAmerican,
		optArrayLen);

	WriteToLog("copyOptionBaseData completed\n");

	copyRateData(
		pGD,
		pHD,
		fDomesticRate, 
		fForeignRate, 
		optArrayLen);

	WriteToLog("copyRateData completed\n");

	
	copyDividendData(
		pGD,
		pHD,
		fSpotPrice,
		divAmounts, 
		divYears, 
		divCount, 
		divIndex, 
		divArrayLen,
		nSteps,
		optArrayLen);

	WriteToLog("copyDividendData completed\n");

	optionsPrepareAndEliminate(
		pGD,
		pHD,
		pHD->h_Volatility, 
		pHD->h_Volatility1, 
		optArrayLen,
		nSteps);

	WriteToLog("optionsPrepareAndEliminate completed\n");

	
	const int ITMAX = 30;
	for(int iter=-1;iter<ITMAX;iter++) 
	{
	  //dumpAllGPUArrays("Before brent step", optArrayLen);

		brentStep(
			pHD->h_a, 
			pHD->h_b, 
			pHD->h_c, 
			pHD->h_d, 
			pHD->h_e, 
			pHD->h_fa, 
			pHD->h_fb, 
			pHD->h_fc, 
			0.01, 
			2.5, 
			pHD->h_CalcFlag,
			iter, 
			optArrayLen, 
			pHD->h_Volatility, 
			pHD->h_Volatility1, 
			pHD->h_rVolatility, 
			fOptionPrice);

	  //dumpAllGPUArrays("After brent step", optArrayLen);

		if(iter>=0) {


			CUDA_SAFE_CALL( cudaMemcpy(pGD->d_CalcFlag,  pHD->h_CalcFlag ,  optArrayLen*sizeof(int), cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL( cudaMemcpy(pGD->d_Volatility,  pHD->h_Volatility ,  optArrayLen*sizeof(float), cudaMemcpyHostToDevice) );

			//dumpAllGPUArrays("Before CalcOpt", optArrayLen);
			calcOpt(pGD, optArrayLen, nSteps);
			//dumpAllGPUArrays("After CalcOpt", optArrayLen);

			CUDA_SAFE_CALL( cudaMemcpy(pHD->h_Volatility,  pGD->d_Volatility ,  optArrayLen*sizeof(float), cudaMemcpyDeviceToHost) );
		}


	}

	//dumpAllGPUArrays("After Calculations", optArrayLen);

	for (int i = 0; i < optArrayLen; ++i)
	{
		fVolatility[i]	= static_cast<float>(pHD->h_rVolatility[i]);
	}

#ifdef _DEBUG
	long nFlag = 0;
	// Compare GPU results with CPU
	for (int idx = 0; idx < optArrayLen; ++idx)
	{
		float dIV = CalcVolatilityMM3(	
			fDomesticRate[idx],											
			fForeignRate[idx],												
			fSpotPrice[idx],													
			fOptionPrice[idx],												
			fStrike[idx],														
			nDTE[idx],																
			nIsCall[idx],											
			nIsAmerican[idx],												
			divCount[idx],
			&(divAmounts[divIndex[idx]]),
			&(divYears[divIndex[idx]]),
			DEFAULT_BINOMIAL_STEPS,
			0,
			0,
			MM_EGAR_BINOMIAL,
			&nFlag);
			


	}

#endif
	return GPU_OK;

}

/*__declspec(dllexport)*/ long GPUCalcVolatility(float*	fDomesticRate,	
																						 float*	fForeignRate, 
																						 float*	fSpotPrice,
																						 float*	fOptionPrice,
																						 float*	fStrike,
																						 int*		nDTE,
																						 int*		nIsCall,
																						 int*		nIsAmerican,
																						 int*		divIndex,
																						 int*		divCount,
																						 float*	divAmounts,
																						 float*	divYears,
																						 long   divArrayLen,
																						 long		optArrayLen,
																						 long		nSteps,
																						 float* fVolatility)
{
	if (!g_isGPUInitialized){
		return GPU_NOT_INITIALIZED;
	}

	//cutLockMutex(h_cs);
	GPU_DEVICE* pGPUDev = selectGPUDevice();
	if (!pGPUDev){
		return GPU_NOT_INITIALIZED;
	}

	WriteToLog("OptArrayLen %d, divArrayLen %d\n", optArrayLen, divArrayLen);

	pGPUDev->curCalcVolatilityTask.DomesticRate					= fDomesticRate;	
	pGPUDev->curCalcVolatilityTask.ForeignRate					= fForeignRate; 
	pGPUDev->curCalcVolatilityTask.SpotPrice						= fSpotPrice;
	pGPUDev->curCalcVolatilityTask.OptionPrice					= fOptionPrice;
	pGPUDev->curCalcVolatilityTask.Strike								= fStrike;
	pGPUDev->curCalcVolatilityTask.DTE									= nDTE;
	pGPUDev->curCalcVolatilityTask.IsCall								= nIsCall;
	pGPUDev->curCalcVolatilityTask.IsAmerican						= nIsAmerican;
	pGPUDev->curCalcVolatilityTask.DivIndex							= divIndex;
	pGPUDev->curCalcVolatilityTask.DivCount							= divCount;
	pGPUDev->curCalcVolatilityTask.DivAmounts						= divAmounts;
	pGPUDev->curCalcVolatilityTask.DivYears							= divYears;
	pGPUDev->curCalcVolatilityTask.DivArrayLen					= divArrayLen;
	pGPUDev->curCalcVolatilityTask.OptArrayLen					= optArrayLen;
	pGPUDev->curCalcVolatilityTask.Steps								= nSteps;
	pGPUDev->curCalcVolatilityTask.Volatility						= fVolatility;		


	pGPUDev->gpuOperationType	= enGPUCalcVolatility;

	cutSetEvent(pGPUDev->hGPUOperationEvent);
	if (!cutTryWaitEvent(pGPUDev->hGPUOperationDoneEvent, MAX_GPU_WAIT_TIMEOUT)){
		WriteToLog("Wait timeout in GPUCalcVolatility. Device ID %d\n", pGPUDev->deviceID);
	}

	cutUnlockMutex(pGPUDev->mutex);


	return pGPUDev->curOperationStatus;
}

long GPUCalcTheoPriceImpl(GPU_DEVICE*	pGPU,
													float*		fDomesticRate,	
													float*		fForeignRate, 
													float*		fSpotPrice,
													float*		fVolatility,
													float*		fStrike,
													int*			nDTE,
													int*			nIsCall,
													int*			nIsAmerican,
													int*			divIndex,
													int*			divCount,
													float*		divAmounts,
													float*		divYears,
													long			divArrayLen,
													long			optArrayLen,
													long			nSteps,
													float*		fTheoPrice)
{

	HOST_DATA*	pHD = &pGPU->HOSTData;
	GPU_DATA*		pGD = &pGPU->GPUData;

	copyOptionBaseData(
		pGD,
		fSpotPrice,
		fStrike,
		fVolatility,
		nDTE,
		nIsCall,
		nIsAmerican,
		optArrayLen);


	copyRateData(
		pGD,
		pHD,
		fDomesticRate, 
		fForeignRate, 
		optArrayLen);

	
	copyDividendData(
		pGD,
		pHD,
		fSpotPrice,
		divAmounts, 
		divYears, 
		divCount, 
		divIndex, 
		divArrayLen,
		nSteps,
		optArrayLen);

	CUDA_SAFE_CALL( cudaMemcpy(pGD->d_Volatility,  fVolatility,   optArrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	
	calcOpt(pGD, optArrayLen, nSteps);


	CUDA_SAFE_CALL( cudaMemcpy(fTheoPrice,  pGD->d_Volatility,   optArrayLen*sizeof(float), cudaMemcpyDeviceToHost) );

	return GPU_OK;	
}

/*__declspec(dllexport)*/ long GPUCalcTheoPrice(float*	fDomesticRate,	
																						float*	fForeignRate, 
																						float*	fSpotPrice,
																						float*	fVolatility,
																						float*	fStrike,
																						int*		nDTE,
																						int*		nIsCall,
																						int*		nIsAmerican,
																						int*		divIndex,
																						int*		divCount,
																						float*	divAmounts,
																						float*	divYears,
																						long    divArrayLen,
																						long		optArrayLen,
																						long		nSteps,
																						float* fTheoPrice)
{
	if (!g_isGPUInitialized){
		return GPU_NOT_INITIALIZED;
	}

	//cutLockMutex(h_cs);
	GPU_DEVICE* pGPUDev = selectGPUDevice();
	if (!pGPUDev){
		return GPU_NOT_INITIALIZED;
	}

	pGPUDev->curTheoPriceTask.DomesticRate					= fDomesticRate;	
	pGPUDev->curTheoPriceTask.ForeignRate					= fForeignRate; 
	pGPUDev->curTheoPriceTask.SpotPrice						= fSpotPrice;
	pGPUDev->curTheoPriceTask.Volatility						= fVolatility;
	pGPUDev->curTheoPriceTask.Strike								= fStrike;
	pGPUDev->curTheoPriceTask.DTE									= nDTE;
	pGPUDev->curTheoPriceTask.IsCall								= nIsCall;
	pGPUDev->curTheoPriceTask.IsAmerican						= nIsAmerican;
	pGPUDev->curTheoPriceTask.DivIndex							= divIndex;
	pGPUDev->curTheoPriceTask.DivCount							= divCount;
	pGPUDev->curTheoPriceTask.DivAmounts						= divAmounts;
	pGPUDev->curTheoPriceTask.DivYears							= divYears;
	pGPUDev->curTheoPriceTask.DivArrayLen					= divArrayLen;
	pGPUDev->curTheoPriceTask.OptArrayLen					= optArrayLen;
	pGPUDev->curTheoPriceTask.Steps								= nSteps;
	pGPUDev->curTheoPriceTask.TheoPrice						= fTheoPrice;


	pGPUDev->gpuOperationType	= enGPUCalcTheoPrice;

	cutSetEvent(pGPUDev->hGPUOperationEvent);
	if (!cutTryWaitEvent(pGPUDev->hGPUOperationDoneEvent, MAX_GPU_WAIT_TIMEOUT)){
		WriteToLog("Wait timeout in GPUCalcTheoPrice. Device ID %d\n", pGPUDev->deviceID);
	}

	
	cutUnlockMutex(pGPUDev->mutex);

	return pGPUDev->curOperationStatus;
}





long GPUCalcGreeksImpl(GPU_DEVICE*	pGPU,
													float*		fDomesticRate,	
													float*		fForeignRate, 
													float*		fSpotPrice,
													float*		fVolatility,
													float*		fStrike,
													int*			nDTE,
													int*			nIsCall,
													int*			nIsAmerican,
													int*			divIndex,
													int*			divCount,
													float*		divAmounts,
													float*		divYears,
													long			divArrayLen,
													long			optArrayLen,
													long			nSteps,
													float*		fGreeks)
{
	HOST_DATA*	pHD = &pGPU->HOSTData;
	GPU_DATA*		pGD = &pGPU->GPUData;

	copyOptionBaseData(
		pGD,
		fSpotPrice,
		fStrike,
		fVolatility,
		nDTE,
		nIsCall,
		nIsAmerican,
		optArrayLen);


	copyRateData(
		pGD,
		pHD,
		fDomesticRate, 
		fForeignRate, 
		optArrayLen);

	
	copyDividendData(
		pGD,
		pHD,
		fSpotPrice,
		divAmounts, 
		divYears, 
		divCount, 
		divIndex, 
		divArrayLen,
		nSteps,
		optArrayLen);


	optionsPrepareAndEliminate(
		pGD,
		pHD,
		pHD->h_Volatility, 
		pHD->h_Volatility1, 
		optArrayLen,
		nSteps);


	memset(fGreeks, 0, optArrayLen*GPU_GREEKS_PER_OPTION*sizeof(float));
	CUDA_SAFE_CALL( cudaMemcpy(pGD->d_GreeksCR1,  fGreeks,   optArrayLen*GPU_GREEKS_PER_OPTION*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGD->d_GreeksCR2,  fGreeks,   optArrayLen*GPU_GREEKS_PER_OPTION*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGD->d_GreeksCR3,  fGreeks,   optArrayLen*GPU_GREEKS_PER_OPTION*sizeof(float), cudaMemcpyHostToDevice) );


	CUDA_SAFE_CALL( cudaMemcpy(pGD->d_theoPriceCR1,  fGreeks,   optArrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGD->d_theoPriceCR2,  fGreeks,   optArrayLen*sizeof(float), cudaMemcpyHostToDevice) );


	memcpy(pHD->h_Volatility, fVolatility, optArrayLen*sizeof(float));
	CUDA_SAFE_CALL( cudaMemcpy(pGD->d_Volatility,  fVolatility,   optArrayLen*sizeof(float), cudaMemcpyHostToDevice) );

	for (int i = 0; i < optArrayLen; ++i){
		pHD->h_Volatility1[i] = pHD->h_Volatility[i] + cdDeltaVolatility;
	}

	calcGreeks(pGD, pHD, optArrayLen, nSteps);

	//set array with shifted rate
	float* fDeltaDomesticRate = pHD->h_Volatility1;
	memcpy(fDeltaDomesticRate, fDomesticRate, optArrayLen*sizeof(float));

	copyRateData(
		pGD,
		pHD,
		fDeltaDomesticRate, 
		fForeignRate, 
		optArrayLen,
		cdDeltaRate);

	copyDividendData(
		pGD,
		pHD,
		fSpotPrice,
		divAmounts, 
		divYears, 
		divCount, 
		divIndex, 
		divArrayLen,
		nSteps,
		optArrayLen);

	calcRhoGreek(pGD, pHD, optArrayLen, nSteps);

	CUDA_SAFE_CALL( cudaMemcpy(pGPU->curCalcGreeksTask.Greeks,  pGD->d_GreeksCR1 ,  
		optArrayLen*GPU_GREEKS_PER_OPTION*sizeof(float), cudaMemcpyDeviceToHost) );


	return GPU_OK;	
}


/*__declspec(dllexport)*/ long GPUCalcGreeks(float*	fDomesticRate,	
																					float*	fForeignRate, 
																					float*	fSpotPrice,
																					float*	fVolatility,
																					float*	fStrike,
																					int*		nDTE,
																					int*		nIsCall,
																					int*		nIsAmerican,
																					int*		divIndex,
																					int*		divCount,
																					float*	divAmounts,
																					float*	divYears,
																					long    divArrayLen,
																					long		optArrayLen,
																					long		nSteps,
																					float*	fGreeks
																					)
{

	if (!g_isGPUInitialized){
		return GPU_NOT_INITIALIZED;
	}

	//cutLockMutex(h_cs);
	GPU_DEVICE* pGPUDev = selectGPUDevice();
	if (!pGPUDev){
		return GPU_NOT_INITIALIZED;
	}

	pGPUDev->curCalcGreeksTask.DomesticRate				= fDomesticRate;	
	pGPUDev->curCalcGreeksTask.ForeignRate				= fForeignRate; 
	pGPUDev->curCalcGreeksTask.SpotPrice					= fSpotPrice;
	pGPUDev->curCalcGreeksTask.Volatility					= fVolatility;
	pGPUDev->curCalcGreeksTask.Strike							= fStrike;
	pGPUDev->curCalcGreeksTask.DTE								= nDTE;
	pGPUDev->curCalcGreeksTask.IsCall							= nIsCall;
	pGPUDev->curCalcGreeksTask.IsAmerican					= nIsAmerican;
	pGPUDev->curCalcGreeksTask.DivIndex						= divIndex;
	pGPUDev->curCalcGreeksTask.DivCount						= divCount;
	pGPUDev->curCalcGreeksTask.DivAmounts					= divAmounts;
	pGPUDev->curCalcGreeksTask.DivYears						= divYears;
	pGPUDev->curCalcGreeksTask.DivArrayLen				= divArrayLen;
	pGPUDev->curCalcGreeksTask.OptArrayLen				= optArrayLen;
	pGPUDev->curCalcGreeksTask.Steps							= nSteps;
	pGPUDev->curCalcGreeksTask.Greeks							= fGreeks;


	pGPUDev->gpuOperationType	= enGPUCalcGreeks;

	cutSetEvent(pGPUDev->hGPUOperationEvent);
	if (!cutTryWaitEvent(pGPUDev->hGPUOperationDoneEvent, MAX_GPU_WAIT_TIMEOUT)){
		WriteToLog("Wait timeout in GPUCalcTheoPrice. Device ID %d\n", pGPUDev->deviceID);
	}
	
	cutUnlockMutex(pGPUDev->mutex);

	return pGPUDev->curOperationStatus;


}
/*__declspec(dllexport)*/ long GPUCalcGreeksCPU(float*	fDomesticRate,	
																					float*	fForeignRate, 
																					float*	fSpotPrice,
																					float*	fVolatility,
																					float*	fStrike,
																					int*		nDTE,
																					int*		nIsCall,
																					int*		nIsAmerican,
																					int*		divIndex,
																					int*		divCount,
																					float*	divAmounts,
																					float*	divYears,
																					long    divArrayLen,
																					long		optArrayLen,
																					long		nSteps,
																					float*	fGreeks
																					)

{
	if (!g_isGPUInitialized){
		return GPU_NOT_INITIALIZED;
	}


	//cutLockMutex(h_cs);


	for(int i = 0; i < optArrayLen; i++) 
	{
		GREEKS greeks;
		greeks.nMask =  GT_ALL;

		CalcGreeksMM2(
			fDomesticRate[i],
			fForeignRate[i],
			fSpotPrice[i],
			fStrike[i],
			fVolatility[i],
			nDTE[i],
			nIsCall[i],											
			nIsAmerican[i],												
			divCount[i],
			&(divAmounts[divIndex[i]]),
			&(divYears[divIndex[i]]),
			nSteps,
			0,
			0,
			MM_EGAR_BINOMIAL,
			&greeks);


		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_THEO_PRICE]				= greeks.dTheoPrice;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_ALPHA]						= greeks.dAlpha;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_DELTA]						= greeks.dDelta;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_GAMMA]						= greeks.dGamma;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_VEGA]							= greeks.dVega;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_THETA]						= greeks.dTheta;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_RHO]							= greeks.dRho;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_DELTA_VEGA]				= greeks.dDeltaVega;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_DELTA_THETA]			= greeks.dDeltaTheta;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_GAMMA_VEGA]				= greeks.dGammaVega;
		fGreeks[i*GPU_GREEKS_PER_OPTION + GPU_GREEK_GAMMA_THETA]			= greeks.dGammaTheta;
		
	}



	return GPU_OK;
}

__host__ void createDataArrays(GPU_DEVICE* pGPU, long arrayLen)
{
	HOST_DATA*	pHD = &pGPU->HOSTData;
	GPU_DATA*		pGD = &pGPU->GPUData;

	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_StockPrice,			arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_StockPrice1,			arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_Strike,					arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_OptionPrice,			arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_IsAmerican,			arrayLen*sizeof(int)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_IsCall,					arrayLen*sizeof(int)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_T,								arrayLen*sizeof(int)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_Volatility,			arrayLen*sizeof(float)) );

	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_theoPriceCR1,		arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_theoPriceCR2,		arrayLen*sizeof(float)) );
		
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_DC,							arrayLen*sizeof(int)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_CalcFlag,				arrayLen*sizeof(int)) );

	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_Rd,							arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_Rf,							arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_ContRd,					arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_ContRf,					arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_Yte,							arrayLen*sizeof(float)) );

	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_Discount,				arrayLen*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_Payment1,				arrayLen*STEPS_LENGTH*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_Payment2,				arrayLen*STEPS_LENGTH*sizeof(float)) );

	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_GreeksCR1,			arrayLen*GPU_GREEKS_PER_OPTION*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_GreeksCR2,			arrayLen*GPU_GREEKS_PER_OPTION*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&pGD->d_GreeksCR3,			arrayLen*GPU_GREEKS_PER_OPTION*sizeof(float)) );

	pHD->h_Yte							= (float*)malloc(arrayLen*sizeof(float));
	pHD->h_ContRd					= (float*)malloc(arrayLen*sizeof(float));
	pHD->h_Discount				= (float*)malloc(arrayLen*sizeof(float));

	pHD->h_Vola1						= (float*)malloc(arrayLen*sizeof(float));
	pHD->h_Vola2						= (float*)malloc(arrayLen*sizeof(float));

	pHD->h_Volatility			= (float*)malloc(arrayLen*sizeof(float));
	pHD->h_Volatility1			= (float*)malloc(arrayLen*sizeof(float));

	pHD->h_a								= (double*)malloc(arrayLen*sizeof(double));
	pHD->h_b								= (double*)malloc(arrayLen*sizeof(double));
	pHD->h_c								= (double*)malloc(arrayLen*sizeof(double));
	pHD->h_d								= (double*)malloc(arrayLen*sizeof(double));
	pHD->h_e								= (double*)malloc(arrayLen*sizeof(double));		
	pHD->h_fa							= (double*)malloc(arrayLen*sizeof(double));
	pHD->h_fb							= (double*)malloc(arrayLen*sizeof(double));
	pHD->h_fc							= (double*)malloc(arrayLen*sizeof(double));

	pHD->h_rVolatility			= (double*)malloc(arrayLen*sizeof(double));

	pHD->h_CalcFlag				= (int*)malloc(arrayLen*sizeof(int));

	memset(pHD->h_Yte, 0, arrayLen*sizeof(float));
	memset(pHD->h_ContRd, 0, arrayLen*sizeof(float));
	memset(pHD->h_Discount, 0, arrayLen*sizeof(float));
	memset(pHD->h_Volatility, 0, arrayLen*sizeof(float));
	memset(pHD->h_Volatility1, 0, arrayLen*sizeof(float));

	memset(pHD->h_a, 0, arrayLen*sizeof(double));
	memset(pHD->h_b, 0, arrayLen*sizeof(double));
	memset(pHD->h_c, 0, arrayLen*sizeof(double));
	memset(pHD->h_d, 0, arrayLen*sizeof(double));
	memset(pHD->h_e, 0, arrayLen*sizeof(double));
	memset(pHD->h_fa, 0, arrayLen*sizeof(double));
	memset(pHD->h_fb, 0, arrayLen*sizeof(double));
	memset(pHD->h_fc, 0, arrayLen*sizeof(double));


	for(int i = 0; i < arrayLen; ++i) {
		pHD->h_Vola1[i]=0.01;
		pHD->h_Vola2[i]=2.5;
	}
}

__host__ void deleteDataArrays(GPU_DEVICE* pGPU)
{
	HOST_DATA*	pHD = &pGPU->HOSTData;
	GPU_DATA*		pGD = &pGPU->GPUData;

	CUDA_SAFE_CALL( cudaFree(pGD->d_StockPrice));
	CUDA_SAFE_CALL( cudaFree(pGD->d_StockPrice1));
	CUDA_SAFE_CALL( cudaFree(pGD->d_Strike));
	CUDA_SAFE_CALL( cudaFree(pGD->d_OptionPrice));
	CUDA_SAFE_CALL( cudaFree(pGD->d_IsAmerican));
	CUDA_SAFE_CALL( cudaFree(pGD->d_IsCall));
	CUDA_SAFE_CALL( cudaFree(pGD->d_T));
	CUDA_SAFE_CALL( cudaFree(pGD->d_Volatility));

	CUDA_SAFE_CALL( cudaFree(pGD->d_theoPriceCR1));
	CUDA_SAFE_CALL( cudaFree(pGD->d_theoPriceCR2));
		
	CUDA_SAFE_CALL( cudaFree(pGD->d_DC));

	CUDA_SAFE_CALL( cudaFree(pGD->d_Rd));
	CUDA_SAFE_CALL( cudaFree(pGD->d_Rf));
	CUDA_SAFE_CALL( cudaFree(pGD->d_ContRd));
	CUDA_SAFE_CALL( cudaFree(pGD->d_ContRf));
	CUDA_SAFE_CALL( cudaFree(pGD->d_Yte));

	CUDA_SAFE_CALL( cudaFree(pGD->d_Discount));
	CUDA_SAFE_CALL( cudaFree(pGD->d_Payment1));
	CUDA_SAFE_CALL( cudaFree(pGD->d_Payment2));

	CUDA_SAFE_CALL( cudaFree(pGD->d_CalcFlag));

	CUDA_SAFE_CALL( cudaFree(pGD->d_GreeksCR1));
	CUDA_SAFE_CALL( cudaFree(pGD->d_GreeksCR2));
	CUDA_SAFE_CALL( cudaFree(pGD->d_GreeksCR3));


	free(pHD->h_Yte);
	free(pHD->h_ContRd);
	free(pHD->h_Discount);
	free(pHD->h_Vola1);
	free(pHD->h_Vola2);

	free(pHD->h_Volatility);
	free(pHD->h_Volatility1);


	free(pHD->h_a);	
	free(pHD->h_b);
	free(pHD->h_c);	
	free(pHD->h_d);	
	free(pHD->h_e);	
	free(pHD->h_fa);
	free(pHD->h_fb);
	free(pHD->h_fc);

	free(pHD->h_rVolatility);
	free(pHD->h_CalcFlag);
}


__host__ void copyOptionBaseData(GPU_DATA*	pGPU,
																 float*			dSpotPrice,
																 float*			dStrike,
																 float*			dOptionPrice,
																 int*				nDTE,
																 int*				nIsCall,
																 int*				nIsAmerican,
																 long				arrayLen)
{
	
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_StockPrice,  dSpotPrice,   arrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Strike,  dStrike,   arrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_OptionPrice,  dOptionPrice ,   arrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_IsAmerican,  nIsAmerican,   arrayLen*sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_IsCall,  nIsCall,   arrayLen*sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_T,  nDTE,   arrayLen*sizeof(int), cudaMemcpyHostToDevice) );

}


__host__ void copyRateData(	GPU_DATA*		pGPU,
														HOST_DATA*	pHD,
														float*			dDomesticRate,	
														float*			dForeignRate, 
														long				arrayLen,
														float				deltaRate)
{
	if (fabs(deltaRate) > 0.0001)
	{
		for (int i = 0; i < arrayLen; ++i)
		{
			dDomesticRate[i] += deltaRate;
		}
	}

	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Rf,  dForeignRate,   arrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Rd,  dDomesticRate ,   arrayLen*sizeof(float), cudaMemcpyHostToDevice) );

	WriteToLog("Prepare rates\n");
	gpuRD2C<<<128,256>>>(pGPU->d_Rd, pGPU->d_Rf, pGPU->d_T, arrayLen, pGPU->d_ContRd, pGPU->d_ContRf, pGPU->d_Yte);

	CUT_CHECK_ERROR("gpuRD2C() execution failed\n");
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	CUDA_SAFE_CALL( cudaMemcpy(pHD->h_ContRd,  pGPU->d_ContRd,   arrayLen*sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(pHD->h_Yte,  pGPU->d_Yte,   arrayLen*sizeof(float), cudaMemcpyDeviceToHost) );

}


__host__ void copyDividendData(		GPU_DATA*		pGPU,
																	HOST_DATA*	pHD,
																	float*		dSpotPrice, 
																	float*		divAmounts, 
																	float*		divYears, 
																	int*			divCount, 
																	int*			divIndex, 
																	long			divArrayLen,
																	long			Steps,
																	long			optArrayLen)
{

	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_DC,  divCount,   optArrayLen*sizeof(int), cudaMemcpyHostToDevice) );

	float* payment1		= (float*)malloc(optArrayLen*STEPS_LENGTH*sizeof(float));
	float* payment2		= (float*)malloc(optArrayLen*STEPS_LENGTH*sizeof(float));

	memset(payment1, 0, optArrayLen*STEPS_LENGTH*sizeof(float));
	memset(payment2, 0, optArrayLen*STEPS_LENGTH*sizeof(float));

	WriteToLog("Prepare dividends\n");
	CachePayments(
		pHD,
		dSpotPrice, 
		pHD->h_Yte, 
		pHD->h_ContRd, 
		divAmounts, 
		divYears, 
		divIndex, 
		divCount, 
		Steps, 
		optArrayLen, 
		pHD->h_Discount, 
		payment1);

	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_StockPrice,  pHD->h_Discount,   optArrayLen*sizeof(float), cudaMemcpyHostToDevice) );

	WriteToLog("Prepare discounts 100\n");
	CachePayments(
		pHD,
		dSpotPrice, 
		pHD->h_Yte, 
		pHD->h_ContRd, 
		divAmounts, 
		divYears, 
		divIndex, 
		divCount, 
		Steps-1, 
		optArrayLen, 
		pHD->h_Discount, 
		payment2);

	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_StockPrice1,  pHD->h_Discount,   optArrayLen*sizeof(float), cudaMemcpyHostToDevice) );

	WriteToLog("Prepare discounts 99\n");

	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Discount,  payment1,   optArrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Payment1,  payment1,   optArrayLen*STEPS_LENGTH*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Payment2,  payment2,   optArrayLen*STEPS_LENGTH*sizeof(float), cudaMemcpyHostToDevice) );

	free(payment1);
	free(payment2);
}

__host__ void CachePayments(HOST_DATA*	pHD,
														float*			stockPrice, 
														float*			dYte, 
														float*			dContRd, 
														float*			DA, 
														float*			DT, 
														int*				DI, 
														int*				DC, 
														long				Steps, 
														long				optCount, 
														float*			dDiscount, 
														float*			Payment) 
{

	for(long opt = 0; opt < optCount; opt ++) {
		pHD->h_Discount[opt] = DividendsDiscount(stockPrice[opt],
			Payment+opt*STEPS_LENGTH, 
			pHD->h_Yte[opt], 
			pHD->h_ContRd[opt], 
			DA+DI[opt], 
			DT+DI[opt], 
			DC[opt], 
			Steps);
	}
}


__host__ float DividendsDiscount(float S, 
																 float* Payment, 
																 float T, 
																 float Rd, 
																 float* DA, 
																 float* DT, 
																 long DC, 
																 long Steps) {

	double S0 = S;

	for (long k = 0; k < DC; k++) 
	{
		if (DT[k] < 0. || !ValueNEQZero(DT[k]) || DT[k] > T || DA[k] < 0. || !ValueNEQZero(DA[k]))
			continue;

		S0 -= DividendPv(DA[k], Rd, DT[k]);						// Discounts the underlying

		long	Step = long(((double)DT[k]) / T  * (double)Steps);	// Calculate payment step

		// Sometimes it can has two dividend payments in one day

		if(Step <= cnGPUTreeStepsMax && Step >= 0L) {
			Payment[Step] += DividendPv(DA[k], Rd, DT[k]);
		}

	}
  double DV = 0.0;
	for (long k = Steps; k >= 0 ; k--) 
	{
		if(Payment[k]>GPU_Epsilon) {
			DV += Payment[k];
			Payment[k] = DV;
		}

	}


//	WriteToLog("DSP: %f  %f\n", S0, S);
	return S0;

}

/*
__host__ float DividendPv(float dDivAmnt, 
								 float dCntRate, 
								 float dTime)
{
	return dDivAmnt * exp(-dCntRate * dTime);
}
*/


__host__ void optionsPrepareAndEliminate(GPU_DATA*	pGPU,
																				 HOST_DATA*	pHD,
																				 float*			hVolatility, 
																				 float*			hVolatility1 , 
																				 long				arrayLen, 
																				 long				Steps)
{

	memcpy(hVolatility1, pHD->h_Vola1, arrayLen*sizeof(float));
	memcpy(hVolatility, pHD->h_Vola2, arrayLen*sizeof(float));

	memset(pHD->h_rVolatility, 0, arrayLen*sizeof(double));
	memset(pHD->h_CalcFlag, 0, arrayLen*sizeof(int));

	
	WriteToLog("Calculate First Step\n");

	//init volas with 0.01
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Volatility,  pHD->h_Volatility1 ,  arrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_CalcFlag,  pHD->h_CalcFlag ,  arrayLen*sizeof(int), cudaMemcpyHostToDevice) );

	WriteToLog("Calculate CR/BS(1)\n");

	calcOpt(pGPU, arrayLen, Steps);

	CUDA_SAFE_CALL( cudaMemcpy(pHD->h_Volatility1,  pGPU->d_Volatility ,  arrayLen*sizeof(float), cudaMemcpyDeviceToHost) );

	//init volas with 2.5
	WriteToLog("Calculate CR/BS(0)\n");
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Volatility,  pHD->h_Volatility ,  arrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_CalcFlag,  pHD->h_CalcFlag ,  arrayLen*sizeof(int), cudaMemcpyHostToDevice) );

	calcOpt(pGPU, arrayLen, Steps);

	CUDA_SAFE_CALL( cudaMemcpy(pHD->h_Volatility,  pGPU->d_Volatility ,  arrayLen*sizeof(float), cudaMemcpyDeviceToHost) );

}

__host__ void calcOpt(GPU_DATA*	pGPU,
											long			arrayLen, 
											int				Steps) {
	 gpuCR<<<64, 201>>>(
		 pGPU->d_StockPrice, 
		 pGPU->d_Strike, 
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_Volatility, 
		 pGPU->d_Yte, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 Steps,	
		 pGPU->d_Payment1, 
		 pGPU->d_DC, 
		 pGPU->d_CalcFlag, 
		 arrayLen, 
		 pGPU->d_theoPriceCR1);

 	 CUT_CHECK_ERROR("gpuCR(1) execution failed\n");
	 CUDA_SAFE_CALL( cudaThreadSynchronize() );

	 //dumpAllGPUArrays("After gpuCR<<<64, 201>>>", arrayLen);


	 gpuCR<<<64, 200>>>(
		 pGPU->d_StockPrice1, 
		 pGPU->d_Strike, 
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_Volatility, 
		 pGPU->d_Yte, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 Steps-1,	
		 pGPU->d_Payment2, 
		 pGPU->d_DC, 
		 pGPU->d_CalcFlag, 
		 arrayLen, 
		 pGPU->d_theoPriceCR2);

 	 CUT_CHECK_ERROR("gpuCR(2) execution failed\n");
	 CUDA_SAFE_CALL( cudaThreadSynchronize() );

	 //dumpAllGPUArrays("After gpuCR<<<64, 200>>>", arrayLen);

	 gpuCREABS<<<128, 256>>>(
		 pGPU->d_ContRd,	
		 pGPU->d_ContRf, 
		 pGPU->d_StockPrice, 
		 pGPU->d_Strike, 
		 pGPU->d_Yte, 
		 pGPU->d_Volatility, 
		 pGPU->d_IsCall, 
		 pGPU->d_IsAmerican, 
		 pGPU->d_OptionPrice, 
		 pGPU->d_DC, 
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_theoPriceCR2, 
		 pGPU->d_CalcFlag, 
		 arrayLen, 
		 pGPU->d_Discount);

 	 CUT_CHECK_ERROR("gpuCREABS() execution failed\n");
	 CUDA_SAFE_CALL( cudaThreadSynchronize() );

	 //dumpAllGPUArrays("After gpuCREABS<<<128, 256>>>", arrayLen);

}

__host__ void calcGreeks(	GPU_DATA*	pGPU,
													HOST_DATA*	pHD,											 
													long			arrayLen, 
													int				Steps)
{
	/*
	dumpGPUArray("pGPU->d_theoPriceCR1 init", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR1 init", pGPU->d_GreeksCR1, 100);

	float* tmp = (float*)malloc(sizeof(float)*arrayLen);
	memset(tmp, 0, sizeof(float)*arrayLen);
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_theoPriceCR1,  tmp ,  arrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_GreeksCR1,  tmp ,  arrayLen*sizeof(float), cudaMemcpyHostToDevice) );

	dumpGPUArray("pGPU->d_theoPriceCR1 zero", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR1 zero", pGPU->d_GreeksCR1, 100);
	*/

	dumpAllGPUArrays(pGPU, pHD, "CalculateGreeks_gpuCR<<<64, 201>>> - 1", 10);

	 CalculateGreeks_gpuCR<<<64, 201>>>(
		 pGPU->d_StockPrice, 
		 pGPU->d_Strike, 
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_Volatility, 
		 pGPU->d_Yte, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 Steps,	
		 pGPU->d_Payment1, 
		 pGPU->d_DC, 
		 arrayLen, 
		 pGPU->d_CalcFlag,
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_GreeksCR3);

 	 CUT_CHECK_ERROR("CalculateGreeks_gpuCR() execution failed\n");
	 CUDA_SAFE_CALL( cudaThreadSynchronize() );

	dumpGPUArray("pGPU->d_theoPriceCR1 after gpuCR<<<64, 201>>>", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR3  after gpuCR<<<64, 201>>>", pGPU->d_GreeksCR3 + 0*GPU_GREEKS_PER_OPTION, 100);

	 CalculateGreeks_gpuCR<<<64, 200>>>(
		 pGPU->d_StockPrice1, 
		 pGPU->d_Strike, 
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_Volatility, 
		 pGPU->d_Yte, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 Steps-1,	
		 pGPU->d_Payment2, 
		 pGPU->d_DC, 
		 arrayLen, 
		 pGPU->d_CalcFlag,
		 pGPU->d_theoPriceCR2, 
		 pGPU->d_GreeksCR2);

 	 CUT_CHECK_ERROR("CalculateGreeks_gpuCR() execution failed\n");
	 CUDA_SAFE_CALL( cudaThreadSynchronize() );

	dumpGPUArray("pGPU->d_theoPriceCR2 after gpuCR<<<64, 200>>>", pGPU->d_theoPriceCR2, 100);
	dumpGPUArray("pGPU->d_GreeksCR2  after gpuCR<<<64, 200>>>", pGPU->d_GreeksCR2 + 0*GPU_GREEKS_PER_OPTION, 100);

	AdjustCR<<<64, 200>>>(
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_theoPriceCR2, 
		 pGPU->d_GreeksCR3, 
		 pGPU->d_GreeksCR2,
		 arrayLen
	);

 	CUT_CHECK_ERROR("AdjustCR() execution failed\n");
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	 AdjustCRBS_FI<<<32, 128>>>(
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_StockPrice, 
		 pGPU->d_Strike, 
		 pGPU->d_Yte, 
		 pGPU->d_Volatility, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 pGPU->d_DC, 
		 pGPU->d_CalcFlag,
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_GreeksCR1,
		 pGPU->d_GreeksCR3, 
		 arrayLen);

	//calculate GPU_GREEK_VEGA, GPU_GREEK_DELTA_VEGA, GPU_GREEK_GAMMA_VEGA

	dumpGPUArray("pGPU->d_theoPriceCR1 after AdjustCRBS_FI<<<32, 128>>>", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR3  after AdjustCRBS_FI<<<32, 128>>>", pGPU->d_GreeksCR3 + 0*GPU_GREEKS_PER_OPTION, 100);


 	CUT_CHECK_ERROR("AdjustGreeksAndTheoPrice() execution failed\n");
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_GreeksCR3,  pGPU->d_GreeksCR1,   GPU_GREEKS_PER_OPTION*arrayLen*sizeof(float), cudaMemcpyDeviceToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Volatility,  pHD->h_Volatility1,   arrayLen*sizeof(float), cudaMemcpyHostToDevice) );

//	dumpAllGPUArrays(pGPU, pHD, "CalculateGreeks_gpuCR<<<64, 201>>> - 2", 2);


	 CalculateGreeks_gpuCR<<<64, 201>>>(
		 pGPU->d_StockPrice, 
		 pGPU->d_Strike, 
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_Volatility, 
		 pGPU->d_Yte, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 Steps,	
		 pGPU->d_Payment1, 
		 pGPU->d_DC, 
		 arrayLen, 
		 pGPU->d_CalcFlag,
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_GreeksCR1);

 	 CUT_CHECK_ERROR("CalculateGreeks_gpuCR() execution failed\n");
	 CUDA_SAFE_CALL( cudaThreadSynchronize() );

	dumpGPUArray("pGPU->d_theoPriceCR1 after CalculateGreeks_gpuCR<<<64, 201>>>", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR1  after CalculateGreeks_gpuCR<<<64, 201>>>", pGPU->d_GreeksCR1 + 0*GPU_GREEKS_PER_OPTION, 100);

	 CalculateGreeks_gpuCR<<<64, 200>>>(
		 pGPU->d_StockPrice1, 
		 pGPU->d_Strike, 
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_Volatility, 
		 pGPU->d_Yte, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 Steps-1,	
		 pGPU->d_Payment2, 
		 pGPU->d_DC, 
		 arrayLen, 
		 pGPU->d_CalcFlag,
		 pGPU->d_theoPriceCR2, 
		 pGPU->d_GreeksCR2);

 	 CUT_CHECK_ERROR("CalculateGreeks_gpuCR() execution failed\n");
	 CUDA_SAFE_CALL( cudaThreadSynchronize() );

	dumpGPUArray("pGPU->d_theoPriceCR1 after CalculateGreeks_gpuCR<<<64, 200>>>", pGPU->d_theoPriceCR2, 100);
	dumpGPUArray("pGPU->d_GreeksCR1  after CalculateGreeks_gpuCR<<<64, 200>>>", pGPU->d_GreeksCR2 + 0*GPU_GREEKS_PER_OPTION, 100);

	AdjustCR<<<64, 200>>>(
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_theoPriceCR2, 
		 pGPU->d_GreeksCR2,
		 pGPU->d_GreeksCR1, 
		 arrayLen
	);

 	CUT_CHECK_ERROR("AdjustCR() execution failed\n");
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	dumpGPUArray("pGPU->d_theoPriceCR1 after AdjustCR<<<64, 200>>>", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR1  after AdjustCR<<<64, 200>>>", pGPU->d_GreeksCR1 + 0*GPU_GREEKS_PER_OPTION, 100);
	dumpGPUArray("pGPU->d_GreeksCR2  after AdjustCR<<<64, 200>>>", pGPU->d_GreeksCR2 + 0*GPU_GREEKS_PER_OPTION, 100);
	dumpGPUArray("pGPU->d_GreeksCR3  after AdjustCR<<<64, 200>>>", pGPU->d_GreeksCR3 + 0*GPU_GREEKS_PER_OPTION, 100);

	 AdjustCRBS_SI<<<32, 128>>>(
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_StockPrice, 
		 pGPU->d_Strike, 
		 pGPU->d_Yte, 
		 pGPU->d_Volatility, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 pGPU->d_DC, 
		 pGPU->d_CalcFlag,
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_GreeksCR1, 
		 pGPU->d_GreeksCR2, 
		 pGPU->d_GreeksCR3, 
		 arrayLen);

 	CUT_CHECK_ERROR("AdjustCRBS_SI() execution failed\n");
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	dumpGPUArray("pGPU->d_theoPriceCR1 after AdjustCRBS_SI<<<64, 200>>>", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR1  after AdjustCRBS_SI<<<64, 200>>>", pGPU->d_GreeksCR1 + 0*GPU_GREEKS_PER_OPTION, 100);

	//////////////////////////////////////////////////////////////////////////
	//calculate GPU_GREEK_RHO
	/////////////////////////////////////////////////////////////////////////

	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_GreeksCR3,  pGPU->d_GreeksCR1,   GPU_GREEKS_PER_OPTION*arrayLen*sizeof(float), cudaMemcpyDeviceToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_Volatility,  pHD->h_Volatility,   arrayLen*sizeof(float), cudaMemcpyHostToDevice) );
	
}

__host__ void calcRhoGreek(	GPU_DATA*	pGPU,
													HOST_DATA*	pHD,											 
													long			arrayLen, 
													int				Steps)
{
	dumpAllGPUArrays(pGPU, pHD, "CalculateGreeks_gpuCR<<<64, 201>>> - 3", 10);




	 CalculateGreeks_gpuCR<<<64, 201>>>(
		 pGPU->d_StockPrice, 
		 pGPU->d_Strike, 
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_Volatility, 
		 pGPU->d_Yte, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 Steps,	
		 pGPU->d_Payment1, 
		 pGPU->d_DC, 
		 arrayLen, 
		 pGPU->d_CalcFlag,
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_GreeksCR1);

 	 CUT_CHECK_ERROR("CalculateGreeks_gpuCR() execution failed\n");
	 CUDA_SAFE_CALL( cudaThreadSynchronize() );

	dumpGPUArray("pGPU->d_theoPriceCR1 after CalculateGreeks_gpuCR<<<64, 201>>>", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR1  after CalculateGreeks_gpuCR<<<64, 201>>>", pGPU->d_GreeksCR1 + 0*GPU_GREEKS_PER_OPTION, 100);

	 CalculateGreeks_gpuCR<<<64, 200>>>(
		 pGPU->d_StockPrice1, 
		 pGPU->d_Strike, 
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_Volatility, 
		 pGPU->d_Yte, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 Steps-1,	
		 pGPU->d_Payment2, 
		 pGPU->d_DC, 
		 arrayLen, 
		 pGPU->d_CalcFlag,
		 pGPU->d_theoPriceCR2, 
		 pGPU->d_GreeksCR2);

 	 CUT_CHECK_ERROR("CalculateGreeks_gpuCR() execution failed\n");
	 CUDA_SAFE_CALL( cudaThreadSynchronize() );

	dumpGPUArray("pGPU->d_theoPriceCR1 after CalculateGreeks_gpuCR<<<64, 200>>>", pGPU->d_theoPriceCR2, 100);
	dumpGPUArray("pGPU->d_GreeksCR1  after CalculateGreeks_gpuCR<<<64, 200>>>", pGPU->d_GreeksCR2 + 0*GPU_GREEKS_PER_OPTION, 100);

	AdjustCR<<<64, 200>>>(
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_theoPriceCR2, 
		 pGPU->d_GreeksCR2,
		 pGPU->d_GreeksCR1, 
		 arrayLen
	);

 	CUT_CHECK_ERROR("AdjustCR() execution failed\n");
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	dumpGPUArray("pGPU->d_theoPriceCR1 after AdjustCR<<<64, 200>>>", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR1  after AdjustCR<<<64, 200>>>", pGPU->d_GreeksCR1 + 0*GPU_GREEKS_PER_OPTION, 100);
	dumpGPUArray("pGPU->d_GreeksCR2  after AdjustCR<<<64, 200>>>", pGPU->d_GreeksCR2 + 0*GPU_GREEKS_PER_OPTION, 100);
	dumpGPUArray("pGPU->d_GreeksCR3  after AdjustCR<<<64, 200>>>", pGPU->d_GreeksCR3 + 0*GPU_GREEKS_PER_OPTION, 100);

	 AdjustCRBS_TI<<<32, 176>>>(
		 pGPU->d_ContRd, 
		 pGPU->d_ContRf, 
		 pGPU->d_StockPrice, 
		 pGPU->d_Strike, 
		 pGPU->d_Yte, 
		 pGPU->d_Volatility, 
		 pGPU->d_IsCall,	
		 pGPU->d_IsAmerican, 
		 pGPU->d_DC, 
		 pGPU->d_CalcFlag,
		 pGPU->d_theoPriceCR1, 
		 pGPU->d_GreeksCR1, 
		 pGPU->d_GreeksCR2, 
		 pGPU->d_GreeksCR3, 
		 arrayLen);

 	CUT_CHECK_ERROR("AdjustCRBS_TI() execution failed\n");
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	dumpGPUArray("pGPU->d_theoPriceCR1 after AdjustCRBS_SI<<<64, 200>>>", pGPU->d_theoPriceCR1, 100);
	dumpGPUArray("pGPU->d_GreeksCR1  after AdjustCRBS_TI<<<64, 200>>>", pGPU->d_GreeksCR1 + 0*GPU_GREEKS_PER_OPTION, 100);
	dumpGPUArray("pGPU->d_GreeksCR2  after AdjustCRBS_TI<<<64, 200>>>", pGPU->d_GreeksCR2 + 0*GPU_GREEKS_PER_OPTION, 100);
	dumpGPUArray("pGPU->d_GreeksCR3  after AdjustCRBS_TI<<<64, 200>>>", pGPU->d_GreeksCR3 + 0*GPU_GREEKS_PER_OPTION, 100);

	CUDA_SAFE_CALL( cudaMemcpy(pGPU->d_GreeksCR1,  pGPU->d_GreeksCR3,   GPU_GREEKS_PER_OPTION*arrayLen*sizeof(float), cudaMemcpyDeviceToDevice) );
}

__host__ void brentStep(double*			a, 
												double*			b, 
												double*			c,
												double*			d,
												double*			e,
												double*			fa,
												double*			fb,
												double*			fc,
												double			x1,  //float x1  = 0.01;
												double			x2,	//float x2  = 2.5;
												int*				calcFlag,
												int iter,
												int optCount,
												float*V,
												float*V1,
												double*rV,
												float*optionPrice) 
{
	const double EPS = 3.0e-7;
	const double tol = 1.0e-6;
	const double tolF = 1.0e-4;

	//Using Brent�s method, find the root of a function func known to lie between x1 and x2.The
	//root, returned as zbrent, will be refined until its accuracy is tol.

	int count = 0;
	for(int opt=0; opt<optCount;opt++) {
		if(rV[opt]>0.01 && rV[opt]<2.5) count++;
		if(calcFlag[opt]<0) continue;

		if(iter==-1) {

			fa[opt] = V1[opt];
			fb[opt] = V[opt];


			if (optionPrice[opt] > (fb[opt] + optionPrice[opt]))
			{
				rV[opt] = V[opt] = -1;
				calcFlag[opt] = -2;
				continue;
			}
			else if (optionPrice[opt] < ((fa[opt] + optionPrice[opt]) + .005))
			{
				rV[opt] = V[opt] = -1;
				calcFlag[opt] = -3;
				continue;
			}


			a[opt]=x1;b[opt]=x2;c[opt]=x2;
			fc[opt]=fb[opt];
			if ((fa[opt] > 0.0 && fb[opt] > 0.0) || (fa[opt] < 0.0 && fb[opt] < 0.0)) {
				//Root must be bracketed in zbrent
				rV[opt] = V[opt] = -1;
				calcFlag[opt] = -4;
			}
		} 
		else
		{
			fb[opt] = V[opt];

			double min1,min2,p,q,r,s,tol1,xm;
			if ((fb[opt] > 0.0 && fc[opt] > 0.0) || (fb[opt] < 0.0 && fc[opt] < 0.0)) {
				c[opt]=a[opt];  //Rename a, b, c and adjust bounding intervald. 
				fc[opt]=fa[opt];
				e[opt]=d[opt]=b[opt]-a[opt];
			}

			if (fabs(fc[opt]) < fabs(fb[opt])) {
				a[opt]=b[opt];
				b[opt]=c[opt];
				c[opt]=a[opt];
				fa[opt]=fb[opt];
				fb[opt]=fc[opt];
				fc[opt]=fa[opt];
			}

			tol1=2.0*EPS*fabs(b[opt])+0.5*tol; //Convergence check.
			xm=0.5*(c[opt]-b[opt]);


			if (fabs(fb[opt]) <= tolF || fabs(xm) <= tol1) {
				rV[opt] = b[opt];
				calcFlag[opt] = -1;
				continue;
			}

			if (fabs(e[opt]) >= tol1 && fabs(fa[opt]) > fabs(fb[opt])) {

				s=fb[opt]/fa[opt]; //Attempt inverse quadratic interpolation.

				if (a[opt] == c[opt]) {
					p=2.0*xm*s;
					q=1.0-s;
				} else {
					q=fa[opt]/fc[opt];
					r=fb[opt]/fc[opt];
					p=s*(2.0*xm*q*(q-r)-(b[opt]-a[opt])*(r-1.0));
					q=(q-1.0)*(r-1.0)*(s-1.0);
				}

				if (p > 0.0) q = -q; //Check whether in bounds.

				p=fabs(p);
				min1=3.0*xm*q-fabs(tol1*q);
				min2=fabs(e[opt]*q);

				if (2.0*p < (min1 < min2 ? min1 : min2)) {
					e[opt]=d[opt]; //Accept interpolation.
					d[opt]=p/q;
				} else {
					d[opt]=xm; //Interpolation failed, use bisection.
					e[opt]=d[opt];
				}
			} else { //Bounds decreasing too slowly, use bisection.
				d[opt]=xm;
				e[opt]=d[opt];
			}

			a[opt]=b[opt]; //Move last best guess to a.
			fa[opt]=fb[opt];

			if (fabs(d[opt]) > tol1) //Evaluate new trial root.
				b[opt] +=d[opt];
			else
				b[opt] += xm>=0.0?tol1:-tol1;

			V[opt] = b[opt];
		}

	}
	if(count>0)
		WriteToLog("opts#: %d %d\n", iter, count);
}



__host__ void dumpGPUArray(const char* name,
														float* p,
														long arrayLen)
{
	float* h_tmp = (float*)malloc(arrayLen*sizeof(float));
	CUDA_SAFE_CALL( cudaMemcpy(h_tmp,  p ,  arrayLen*sizeof(float), cudaMemcpyDeviceToHost) );


	char szBuf[4096];
	sprintf(szBuf, "%s: ", name);
	for (long i = 0; i < arrayLen; ++i)
	{
		sprintf(szBuf + strlen(szBuf), "%f ", h_tmp[i]);			
	}

	free(h_tmp);

	//WriteToLog("%s\n", szBuf);
	FILE* f = fopen("mparams.txt", "a");
	fprintf(f, "%s\n", szBuf);
	fclose(f);

}

__host__ void dumpNativeArray(const char* name,
														double* p,
														long arrayLen)
{
	char szBuf[8192];
	sprintf(szBuf, "%s: ", name);
	for (long i = 0; i < arrayLen; ++i)
	{
		sprintf(szBuf + strlen(szBuf), "%f ", p[i]);			
	}
	//WriteToLog("%s\n", szBuf);

	FILE* f = fopen("mparams.txt", "a");
	fprintf(f, "%s\n", szBuf);
	fclose(f);

}

__host__ void dumpNativeArray(const char* name,
														float* p,
														long arrayLen)
{
	char szBuf[8192];
	sprintf(szBuf, "%s: ", name);
	for (long i = 0; i < arrayLen; ++i)
	{
		sprintf(szBuf + strlen(szBuf), "%f ", p[i]);			
	}
	//WriteToLog("%s\n", szBuf);

	FILE* f = fopen("mparams.txt", "a");
	fprintf(f, "%s\n", szBuf);
	fclose(f);

}

__host__ void dumpNativeArray(const char* name,
														int* p,
														long arrayLen)
{
	char szBuf[8192];
	sprintf(szBuf, "%s: ", name);
	for (long i = 0; i < arrayLen; ++i)
	{
		sprintf(szBuf + strlen(szBuf), "%d ", p[i]);			
	}
	//WriteToLog("%s\n", szBuf);

	FILE* f = fopen("mparams.txt", "a");
	fprintf(f, "%s\n", szBuf);
	fclose(f);
}

__host__ void dumpGPUArray(const char* name,
														int* p,
														long arrayLen)
{

	int* h_tmp = (int*)malloc(arrayLen*sizeof(int));
	CUDA_SAFE_CALL( cudaMemcpy(h_tmp,  p ,  arrayLen*sizeof(int), cudaMemcpyDeviceToHost) );


	char szBuf[4096];
	sprintf(szBuf, "%s: ", name);
	for (long i = 0; i < arrayLen; ++i)
	{
		sprintf(szBuf + strlen(szBuf), "%d ", h_tmp[i]);			
	}

	free(h_tmp);

	//WriteToLog("%s\n", szBuf);
	FILE* f = fopen("mparams.txt", "a");
	fprintf(f, "%s\n", szBuf);
	fclose(f);

}


__host__ void dumpAllGPUArrays(GPU_DATA*		pGPU,
															 HOST_DATA*		pHD,
															 const char*	name, 
															 long					optArrayLen)
{
	FILE* f = fopen("mparams.txt", "a");
	fprintf(f, "%s\n", name);
	fclose(f);

	dumpGPUArray("pGPU->d_StockPrice", pGPU->d_StockPrice, optArrayLen);
	dumpGPUArray("pGPU->d_StockPrice1", pGPU->d_StockPrice1, optArrayLen);
	dumpGPUArray("pGPU->d_Strike", pGPU->d_Strike, optArrayLen);
	dumpGPUArray("pGPU->d_OptionPrice", pGPU->d_OptionPrice, optArrayLen);
	dumpGPUArray("pGPU->d_IsAmerican", pGPU->d_IsAmerican, optArrayLen);
	dumpGPUArray("pGPU->d_IsCall", pGPU->d_IsCall, optArrayLen);
	dumpGPUArray("pGPU->d_T", pGPU->d_T, optArrayLen);
	dumpGPUArray("pGPU->d_Rd", pGPU->d_Rd, optArrayLen);
	dumpGPUArray("pGPU->d_Rf", pGPU->d_Rf, optArrayLen);
	dumpGPUArray("pGPU->d_ContRd", pGPU->d_ContRd, optArrayLen);
	dumpGPUArray("pGPU->d_ContRf", pGPU->d_ContRf, optArrayLen);
	dumpGPUArray("pGPU->d_Yte", pGPU->d_Yte, optArrayLen);
	dumpGPUArray("pGPU->d_DC", pGPU->d_DC, optArrayLen);
	dumpGPUArray("pGPU->d_Volatility", pGPU->d_Volatility, optArrayLen);
	dumpGPUArray("pGPU->d_Discount", pGPU->d_Discount, optArrayLen);
	dumpGPUArray("pGPU->d_Payment1", pGPU->d_Payment1, optArrayLen);
	dumpGPUArray("pGPU->d_Payment2", pGPU->d_Payment2, optArrayLen);
	dumpGPUArray("pGPU->d_theoPriceCR1", pGPU->d_theoPriceCR1, optArrayLen);
	dumpGPUArray("pGPU->d_theoPriceCR2", pGPU->d_theoPriceCR2, optArrayLen);
	dumpGPUArray("pGPU->d_CalcFlag", pGPU->d_CalcFlag, optArrayLen);

	dumpNativeArray("h_Volatility", pHD->h_Volatility, optArrayLen);
	dumpNativeArray("h_Volatility1", pHD->h_Volatility1, optArrayLen);
	dumpNativeArray("h_rVolatility", pHD->h_rVolatility, optArrayLen);

}



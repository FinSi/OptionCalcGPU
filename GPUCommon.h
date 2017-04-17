#ifndef __GPU_COMMON_H__
#define __GPU_COMMON_H__

#include "GPUThread.h"

#ifdef __cplusplus
extern "C" {
#endif


struct GPU_DATA
{
	////////////////////////////////////////////////////
	// GPU arrays
	////////////////////////////////////////////////////
	float*	d_StockPrice;
	float*	d_StockPrice1;
	float*	d_Strike;
	float*	d_OptionPrice;
	int*		d_IsAmerican;
	int*		d_IsCall;
	int*		d_T;  
	float*	d_Rd;  
	float*	d_Rf;  
	float*	d_ContRd;  
	float*	d_ContRf;  
	float*	d_Yte;
	int*		d_DC;
	float*	d_Volatility;
	float*	d_Discount;
	float*	d_Payment1;
	float*	d_Payment2;
	float*	d_theoPriceCR1;
	float*	d_theoPriceCR2;
	int*		d_CalcFlag;  

	float*	d_GreeksCR1;
	float*	d_GreeksCR2;
	float*	d_GreeksCR3;
};

struct HOST_DATA
{
	////////////////////////////////////
	//// Host arraya
	/////////////////////////////////////
	float* h_Yte;
	float* h_ContRd;
	float* h_Discount;
	float* h_Vola1;
	float* h_Vola2;
	float* h_Volatility;
	float* h_Volatility1;
	int*	 h_CalcFlag;  

	double* h_a; 
	double* h_b;
	double* h_c;
	double* h_d;
	double* h_e;
	double* h_fa;
	double* h_fb;
	double* h_fc;

	double* h_rVolatility;
};

struct CALC_VOLATILITY_TASK
{
	float*	DomesticRate;	
	float*	ForeignRate; 
	float*	SpotPrice;
	float*	OptionPrice;
	float*	Strike;
	int*		DTE;
	int*		IsCall;
	int*		IsAmerican;
	int*		DivIndex;
	int*		DivCount;
	float*	DivAmounts;
	float*	DivYears;
	long		DivArrayLen;
	long		OptArrayLen;
	long		Steps;
	float*	Volatility;		
};

struct CALC_THEO_PRICE_TASK
{
	float*	DomesticRate;	
	float*	ForeignRate; 
	float*	SpotPrice;
	float*	Volatility;
	float*	Strike;
	int*		DTE;
	int*		IsCall;
	int*		IsAmerican;
	int*		DivIndex;
	int*		DivCount;
	float*	DivAmounts;
	float*	DivYears;
	long    DivArrayLen;
	long		OptArrayLen;
	long		Steps;
	float*	TheoPrice;
};

struct CALC_GREEKS_TASK
{
	float*	DomesticRate;	
	float*	ForeignRate; 
	float*	SpotPrice;
	float*	Volatility;
	float*	Strike;
	int*		DTE;
	int*		IsCall;
	int*		IsAmerican;
	int*		DivIndex;
	int*		DivCount;
	float*	DivAmounts;
	float*	DivYears;
	long    DivArrayLen;
	long		OptArrayLen;
	long		Steps;
	float*	Greeks;
};

enum GPUOperationType
{
	enGPUInit,
	enGPUCalcVolatility,
	enGPUCalcTheoPrice,
	enGPUCalcGreeks,
	enGPUClose
};

struct GPU_DEVICE
{
	GPU_DATA					GPUData;
	HOST_DATA					HOSTData;


	GPUMutex					mutex;

	//CRITICAL_SECTION* cs;

	GPUThread					hGPUWorkingThread;

	GPUEvent					hGPUOperationEvent;
	GPUEvent					hGPUOperationDoneEvent;

	GPUOperationType	gpuOperationType;

	CALC_VOLATILITY_TASK	curCalcVolatilityTask;
	CALC_THEO_PRICE_TASK	curTheoPriceTask;
	CALC_GREEKS_TASK			curCalcGreeksTask;
	long curOperationStatus;
	long deviceID;
};

void GPUWorkingThread(void* pData);

GPU_DEVICE* selectGPUDevice();


long GPUInitImpl(GPU_DEVICE* pGPU);
long GPUCloseImpl(GPU_DEVICE* pGPU);
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
													 float*				fVolatility);

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
													float*		fTheoPrice);

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
													float*		fGreeks);


extern GPU_DEVICE* g_pGPUList;
extern int g_nDeviceCount;
extern long g_isGPUInitialized;


#ifdef __cplusplus
} // extern "C"
#endif

#endif


#include "stdafx.h"
#include "GPUCommon.h"
#include "common.h"
#include "GPUThread.h"
#include <zthread/Thread.h>



void GPUWorkingThread(void* pData)
{
	WriteToLog("GPUWorkingThread started\n");

	GPU_DEVICE* pGPUDevice = (GPU_DEVICE*)pData;

	bool isWorkCompleted = false;
	while (!isWorkCompleted)
	{

		cutWaitEvent(pGPUDevice->hGPUOperationEvent);
		
		switch(pGPUDevice->gpuOperationType)
		{
			case enGPUInit:
				WriteToLog("Processing GPUInit request. Device ID %d\n", pGPUDevice->deviceID);
				pGPUDevice->curOperationStatus = GPUInitImpl(pGPUDevice);
				WriteToLog("GPUInit request processed. Device ID %d\n", pGPUDevice->deviceID);
				cutSetEvent(pGPUDevice->hGPUOperationDoneEvent);
				break;
			case enGPUCalcVolatility:
				WriteToLog("Processing GPUCalcVolatility request. Device ID %d\n", pGPUDevice->deviceID);
				pGPUDevice->curOperationStatus = GPUCalcVolatilityImpl(pGPUDevice,
					pGPUDevice->curCalcVolatilityTask.DomesticRate,
					pGPUDevice->curCalcVolatilityTask.ForeignRate,	
					pGPUDevice->curCalcVolatilityTask.SpotPrice,		
					pGPUDevice->curCalcVolatilityTask.OptionPrice,	
					pGPUDevice->curCalcVolatilityTask.Strike,			
					pGPUDevice->curCalcVolatilityTask.DTE,					
					pGPUDevice->curCalcVolatilityTask.IsCall,			
					pGPUDevice->curCalcVolatilityTask.IsAmerican,	
					pGPUDevice->curCalcVolatilityTask.DivIndex,		
					pGPUDevice->curCalcVolatilityTask.DivCount,		
					pGPUDevice->curCalcVolatilityTask.DivAmounts,	
					pGPUDevice->curCalcVolatilityTask.DivYears,		
					pGPUDevice->curCalcVolatilityTask.DivArrayLen,	
					pGPUDevice->curCalcVolatilityTask.OptArrayLen,	
					pGPUDevice->curCalcVolatilityTask.Steps,				
					pGPUDevice->curCalcVolatilityTask.Volatility);

				WriteToLog("GPUCalcVolatility request processed. Device ID %d\n", pGPUDevice->deviceID);
				cutSetEvent(pGPUDevice->hGPUOperationDoneEvent);
				break;

			case enGPUCalcTheoPrice:

				WriteToLog("Processing GPUCalcTheoPrice request. Device ID %d\n", pGPUDevice->deviceID);
				pGPUDevice->curOperationStatus = GPUCalcTheoPriceImpl(pGPUDevice,
					pGPUDevice->curTheoPriceTask.DomesticRate,
					pGPUDevice->curTheoPriceTask.ForeignRate,	
					pGPUDevice->curTheoPriceTask.SpotPrice,		
					pGPUDevice->curTheoPriceTask.Volatility,		
					pGPUDevice->curTheoPriceTask.Strike,				
					pGPUDevice->curTheoPriceTask.DTE,					
					pGPUDevice->curTheoPriceTask.IsCall,				
					pGPUDevice->curTheoPriceTask.IsAmerican,		
					pGPUDevice->curTheoPriceTask.DivIndex,			
					pGPUDevice->curTheoPriceTask.DivCount,			
					pGPUDevice->curTheoPriceTask.DivAmounts,		
					pGPUDevice->curTheoPriceTask.DivYears,			
					pGPUDevice->curTheoPriceTask.DivArrayLen,	
					pGPUDevice->curTheoPriceTask.OptArrayLen,	
					pGPUDevice->curTheoPriceTask.Steps,				
					pGPUDevice->curTheoPriceTask.TheoPrice);

				WriteToLog("GPUCalcTheoPrice request processed. Device ID %d\n", pGPUDevice->deviceID);
				cutSetEvent(pGPUDevice->hGPUOperationDoneEvent);
				break;

			case enGPUCalcGreeks:

				WriteToLog("Processing GPUCalcGreeks request. Device ID %d\n", pGPUDevice->deviceID);
				pGPUDevice->curOperationStatus = GPUCalcGreeksImpl(pGPUDevice,
					pGPUDevice->curCalcGreeksTask.DomesticRate,
					pGPUDevice->curCalcGreeksTask.ForeignRate,	
					pGPUDevice->curCalcGreeksTask.SpotPrice,		
					pGPUDevice->curCalcGreeksTask.Volatility,		
					pGPUDevice->curCalcGreeksTask.Strike,				
					pGPUDevice->curCalcGreeksTask.DTE,					
					pGPUDevice->curCalcGreeksTask.IsCall,				
					pGPUDevice->curCalcGreeksTask.IsAmerican,		
					pGPUDevice->curCalcGreeksTask.DivIndex,			
					pGPUDevice->curCalcGreeksTask.DivCount,			
					pGPUDevice->curCalcGreeksTask.DivAmounts,		
					pGPUDevice->curCalcGreeksTask.DivYears,			
					pGPUDevice->curCalcGreeksTask.DivArrayLen,	
					pGPUDevice->curCalcGreeksTask.OptArrayLen,	
					pGPUDevice->curCalcGreeksTask.Steps,				
					pGPUDevice->curCalcGreeksTask.Greeks);

				WriteToLog("GPUCalcGreeks request processed. Device ID %d\n", pGPUDevice->deviceID);
				cutSetEvent(pGPUDevice->hGPUOperationDoneEvent);
				break;

			case enGPUClose:
				WriteToLog("Processing GPUClose request. Device ID %d\n", pGPUDevice->deviceID);
				pGPUDevice->curOperationStatus = GPUCloseImpl(pGPUDevice);
				WriteToLog("GPUClose request processed. Device ID %d\n", pGPUDevice->deviceID);
				isWorkCompleted = true;
				cutSetEvent(pGPUDevice->hGPUOperationDoneEvent);
				break;
			default:
				WriteToLog("Invalid request id %d in GPUWorkingThread\n", pGPUDevice->gpuOperationType);

		}

	}

	WriteToLog("GPUWorkingThread completed\n");
}

GPU_DEVICE* selectGPUDevice()
{

	while(1){
		for (int i = 0; i < g_nDeviceCount; ++i)
		{	
			if (cutTryLockMutex(g_pGPUList[i].mutex, 0)){
				return &g_pGPUList[i];
			}
		}
		WriteToLog("Can't select GPU device. All GPU devices are busy\n");
		ZThread::Thread::sleep(100);
	}
	return NULL;
}

#pragma once


extern "C"{

	//GPU status codes
	static const long GPU_OK							= 0;
	static const long GPU_NOT_INITIALIZED = 1;
	static const long GPU_ERROR						= 2;
	static const long GPU_TIMEOUT					= 3;

	//GPU greeks constants
	static const long GPU_GREEKS_PER_OPTION					= 11;
	static const long GPU_GREEK_THEO_PRICE					=	0;
	static const long GPU_GREEK_ALPHA								=	1;
	static const long GPU_GREEK_DELTA								=	2;
	static const long GPU_GREEK_GAMMA								=	3;
	static const long GPU_GREEK_VEGA								=	4;
	static const long GPU_GREEK_THETA								=	5;
	static const long GPU_GREEK_RHO									=	6;
	static const long GPU_GREEK_DELTA_VEGA					=	7;
	static const long GPU_GREEK_DELTA_THETA					=	8;
	static const long GPU_GREEK_GAMMA_VEGA					=	9;
	static const long GPU_GREEK_GAMMA_THETA					=	10;


	/*__declspec(dllexport)*/ long GPUInit();
	/*__declspec(dllexport)*/ long GPUClose();
 

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
		float* fVolatility);

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
		float* fTheoPrice);



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
		);

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
		);

}


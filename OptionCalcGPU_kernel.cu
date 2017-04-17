
#ifndef _COXROSS_KERNEL_H_
#define _COXROSS_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <malloc.h>
#include <memory.h>

//#include <float.h>

#ifndef FLT_EPSILON
#define FLT_EPSILON     1.192092896e-07F        /* smallest such that 1.0+FLT_EPSILON != 1.0 */
#endif

enum GPU_VOLATILITY_FLAG
{
	GPU_VF_OK	        = 0x00000000,
	GPU_VF_TOOLOW       = 0x00000001,
	GPU_VF_TOOHIGH      = 0x00000002,
	GPU_VF_UNRELIABLE	= 0x00000003
};

static const __device__ float cdGPUDaysPerYear360 = 360.0;

static const __device__ float cdGPUDaysPerYear365 = 365.0;

static const __device__ float badGPUFloatValue = -1E+38;

//=============================================================================
// Pi constant value
//=============================================================================
static const __device__ float GPU_Pi = 3.1415926535897932384626433832795;

//=============================================================================
// Epsilon constant value
//=============================================================================
static const __device__ float GPU_Epsilon = 0.000001;

//=============================================================================
// Bad float value
//=============================================================================
static const __device__ float GPU_BadfloatValue = -1E+37;

//=============================================================================
// Max theo price value
//=============================================================================
static const __device__ float GPU_MaxTheoPriceValue = 10000.;

//=============================================================================
// Bad theo price value
//=============================================================================
static const __device__ float GPUBadTheoPrice = -1.0;

//=============================================================================
// Epsilon constant value
//=============================================================================
static const __device__ float GPUEpsilon = 0.00001;

//=============================================================================
// Max theo price value
//=============================================================================
static const __device__ float GPUMaxTheoPriceValue = 10000.0;

//=============================================================================
// Max. and min. steps of binomial tree 
//=============================================================================

static const __device__ long cnGPUTreeStepsMax = 100;
static const __device__ long cnGPUTreeStepsMin = 5;

static const __device__ float cdGPUDeltaVolatility = 0.01;

static const __device__ float cdGPUDeltaRate = 0.01;

static const __device__ float cdGPUDeltaTime = 1. / 365.;

static const __device__ float cdGPUDeltaSqrtTime = 0.0523;

/*
#define MAX_OPT 300000
#define STEPS_LENGTH 101

static float* dStockPrice		;
static float* dStockPrice1	;
static float* dStrike				;
static float* dOptionPrice	;
static int* dIsAmerican			;
static int* dIsCall					;
static int* dT							;  
static float* dRd						;  
static float* dRf						;  
static float* dContRd				;  
static float* dContRf				;  
static float* dYte					;
static int* dDC							;
static float* dVolatility		;
static float* dDiscount			;
static float* dPayment1			;
static float* dPayment2			;
static int* dCalcFlag  			;
static float* theoPriceCR1  ;
static float* theoPriceCR2  ;
*/


__device__ float DividendPv_device(float dDivAmnt, float dCntRate, float dTime)
{
	return dDivAmnt * exp(-dCntRate * dTime);
}

__device__ bool IsBadValue_device(float dVal)
{

	if (isnan(dVal) || !isfinite(dVal))
		return false;

	return !(dVal > badGPUFloatValue);
}

__device__ bool ValueNEQZero_device(float dVal)
{
	return fabs(dVal) > FLT_EPSILON;
}

__device__ bool IsValidTheoPrice_device(float dTheoPrice)
{

	if (isnan(dTheoPrice) || !isfinite(dTheoPrice))
		return false;

	if (dTheoPrice < -FLT_EPSILON || dTheoPrice > GPU_MaxTheoPriceValue)
		return false;

	// set -DBL_EPSILON to 0. (price cann't be negative)
	if(!ValueNEQZero_device(dTheoPrice))
		dTheoPrice = 0.;

	return true;	
}

#define A1 0.31938153f
#define A2 -0.356563782f
#define A3 1.781477937f
#define A4 -1.821255978f
#define A5 1.330274429f
#define RSQRT2PI 0.3989422804f



__device__ float  NormalC_device(float d)
{

	float
		K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

	float
		cnd = RSQRT2PI * expf(- 0.5f * d * d) * 
		(K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

	if(d > 0)
		cnd = 1.0f - cnd;

	return cnd;


}

__device__ float BlackAndScholes_device(	
																				float dContRd,
																				float dContRf,
																				float dDiscountedSpot,//dSpotPrice,
																				float dStrike,
																				float dYte,
																				float dVolatility,
																				int	bIsCall)
{

	

	const float A = dDiscountedSpot * (1. / exp( dContRf * dYte));
	const float B = dStrike * (1. / exp( dContRd * dYte));


	const float D = dVolatility * sqrt(dYte);


	const float d1	= log(dDiscountedSpot/dStrike) / D + D * (((dContRd - dContRf) / (dVolatility * dVolatility)) + .5);

	const float d2	= d1 - D;


	const float N1	= NormalC_device(d1);
	const float N2	= NormalC_device(d2);

	float dTheoPrice = bIsCall ? A * N1 - B * N2 : A * ( N1 - 1.) - B * ( N2 - 1.);

	if(!IsValidTheoPrice_device(dTheoPrice))
	{
		dTheoPrice = GPU_BadfloatValue;
	}

	return  dTheoPrice;
}



#define sh(l) k+2*l

__device__ float   
coxRossGPU_device(
									const float* S,		// Discounted underlying spot price
									const float* K,		// Strike price
									const float Rd,	// Domestic continuos risk free rate
									float Rf,	// Foreign continuos risk free rate (or yield value)
									const float* V,		// Volatility
									const float  T,		// Years amount till expiration
									const bool IsCall,	// true if it's 'Call option', false is 'Put option'
									const int	Steps,// Amount of binomial tree steps		
									float* DP,
									//int DC, //Dividends payments
									int opt
									)
{

	// Validate the input parametrs
	if (Steps < cnGPUTreeStepsMin || Steps > cnGPUTreeStepsMax) {
		return badGPUFloatValue;
	}

	if (S[opt] <= 0. || K[opt] <= 0. || V[opt] <= 0. || T <= 0. || Rd < 0. || Rf < 0.) {
		return badGPUFloatValue;
	}

	if (DP != NULL)	
		Rf = 0.;

	__shared__ float	TreeA[cnGPUTreeStepsMax + cnGPUTreeStepsMax + 1];
	__shared__ float	_Tree[cnGPUTreeStepsMax + cnGPUTreeStepsMax + 1];			
	__shared__ float	expc[cnGPUTreeStepsMax + cnGPUTreeStepsMax + 1];			
  __shared__ float	dpv[cnGPUTreeStepsMax + 1]; 
	
	for(int i = threadIdx.x; i < STEPS_LENGTH; i += blockDim.x) {
		dpv[i] = DP==NULL?0:DP[i];
	}

	const float S0 = S[opt];

	const float _K = K[opt]; 
	const float	R = Rd - Rf;							// Summary rate of underlying
	const float	A = expf(R * T / Steps);					// Yield on one step 
	const float	U = V[opt] * sqrtf(T / Steps);				
	const float	D = -U;
	const float	P = (A - expf(D)) / (expf(U) - expf(D)); 

	// Probabilities of changing spot price 
	const float	Pu = P / expf(Rd * T / Steps);		// Upper
	const float	Pd = (1-P) / expf(Rd * T / Steps);	// Lower

	// Set start values of the option prices at the last step
	for(int i = threadIdx.x; i <= 2*Steps; i += blockDim.x){
		const float _S	= S0 * (expc[i] = expf((!IsCall ? -U : U) * (Steps - i)));
		const float C	= (IsCall ? _S - _K : _K - _S);

		TreeA[i] = _Tree[i] = fmaxf(C, 0.0f);
	}

	for(int k = 1; k <= Steps ;){
		__syncthreads();
		for(int l = threadIdx.x; l <= Steps - k; l += blockDim.x) {
			
			float L = TreeA[sh(l)-1];
			float R = TreeA[sh(l)+1];
			
			if(dpv[Steps - k + 1]>FLT_EPSILON && DP!=NULL) {
				const float _S =S0 + dpv[Steps - k + 1];
				const float S1 = _S * expc[sh(l)-1];
				const float S2 = _S * expc[sh(l)+1];
				const float S3 = _S * expc[sh(l)];
				L = fmaxf(IsCall ? S1 - _K : _K - S1, L);
				R = fmaxf(_Tree[sh(l)+1] = fmaxf(IsCall ? S2 - _K : _K - S2, 0.0), R);
				_Tree[sh(l)] =  fmaxf(IsCall ? S3 - _K : _K - S3, 0.0);
				
			}
			TreeA[sh(l)] = !IsCall ?
				fmaxf(Pu * R + Pd * L, _Tree[sh(l)]):
				fmaxf(Pu * L + Pd * R, _Tree[sh(l)]);
				

		}
		k++;

	}


	__syncthreads();

	float dTheoPrice = TreeA[Steps]; // return option value from top of the tree

	if(!IsValidTheoPrice_device(dTheoPrice))
	{
		dTheoPrice = GPUBadTheoPrice;
	}

	return dTheoPrice;
}


__device__ float DiscountForDividends_device(
	float*	S, 
	float	R,
	float*		pdDivAmnt,
	float*		pdDivYears,
	int* nDivCount,
	float	T,
	int opt)
{

	float SumPV = 0;

	for (unsigned n = 0; n < nDivCount[opt]; n++) 
	{
		bool IsGood = (pdDivYears[n] >= 0 && pdDivYears[n] <= T && pdDivAmnt[n]>0);

		if (IsGood) 
		{
			SumPV += DividendPv_device(pdDivAmnt[n], R, pdDivYears[n]);

			if (SumPV > S[opt])
				return GPU_BadfloatValue;
		}
	}

	return S[opt] - SumPV;
}


__device__ float RateDiscToCont_device(float dDiscRate, unsigned int nDays) 
{
	float dYears360 = nDays / 360.0;
	float dYears365 = nDays / 365.0;

	float dDF;

	if (dYears360 <= 1.)
	{
		dDF = 1. + dDiscRate * dYears360;		
	}
	else
	{
		dDF = powf(1. + dDiscRate, dYears360);		
	}

	return log(dDF) / dYears365;
}

// Prepare rates dicount and daystoexp
__global__ void gpuRD2C(float* Rd, float* Rf,  int* T, int optCount, float* dContRd, float* dContRf, float* dYte) {

	const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	for(int opt = tid; opt < optCount; opt += THREAD_N) {
		dContRd[opt] = RateDiscToCont_device(Rd[opt] , T[opt]);
		dContRf[opt] = RateDiscToCont_device(Rf[opt] , T[opt]);
		dYte[opt] = T[opt]/365.0;
	}

}

// Calculate CR

__global__ void gpuCR(
					 float*	S,		// Underlying spot price
					 float*	K,		// Strike price
					 float*	dContRd,		// Domestic continuos risk free rate
					 float* dContRf,		// Foreign continuos risk free rate (or yield value)
					 float* V,		// Volatility
					 float* dYte,		// Years amount till expiration
					 int* IsCall,	// true if it's 'Call option', false is 'Put option'
					 int* IsAmerican,	// true if it's 'Call option', false is 'Put option'
					 int	Steps,	// Amount of binomial tree steps		
					 float*	DP,		// dividend payments)
					 int*	DC,		// count of payment)
					 int* dCalcFlag,
					 int optCount,
					 float* theoPrice//out
					 )
{
	
	for(int opt = blockIdx.x; opt < optCount; opt += gridDim.x) {
		if(IsAmerican[opt] && (!IsCall[opt] || DC[opt] > 0) && dCalcFlag[opt]>-1)
			theoPrice[opt] = coxRossGPU_device(
			S,		// Underlying spot price
			K,		// Strike price
			dContRd[opt],		// Domestic continuos risk free rate
			dContRf[opt],		// Foreign continuos risk free rate (or yield value)
			V,		// Volatility
			dYte[opt],		// Years amount till expiration
			IsCall[opt]!=0,	// true if it's 'Call option', false is 'Put option'
			Steps,
			DC[opt]>0?DP+opt*STEPS_LENGTH:NULL,
			opt//out/
			);
	}
}


// Calculculate BS
__global__ void gpuBS(											
									 float* dContRd,
									 float* dContRf,
									 float* dDiscountedSpot,//Discounted spot price,
									 float* dStrike,
									 float* dYte,
									 float* dVolatility,
									 int*	bIsCall,
									 int optCount,
									 float* theoPrice) {

	const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	for(int opt = tid; opt < optCount; opt += THREAD_N) {
		if(dVolatility[opt]>-1)
			theoPrice[opt] = BlackAndScholes_device(dContRd[opt], dContRf[opt], dDiscountedSpot[opt], dStrike[opt], dYte[opt], dVolatility[opt], bIsCall[opt]);
	}

}

__global__ void gpuCREABS(					
									 float* dContRd,
									 float* dContRf,
									 float* dDiscountedSpot,//Discounted spot price,
									 float* dStrike,
									 float* dYte,
									 float* dVolatility,
									 int*	bIsCall,
									 int*	bIsAmerican,
									 float* dOptionPrice,
									 int* DC,
									 float* theoPriceCR1,
									 float* theoPriceCR2,
									 int* dCalcFlag,
									 int optCount,
									 float* pBlackAndScholes
) {

	const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	for(int opt = tid; opt < optCount; opt += THREAD_N) {
		if(dOptionPrice[opt]>0.001 && dCalcFlag[opt]>-1) {
			
			float p = BlackAndScholes_device(dContRd[opt], DC[opt]>0?0:dContRf[opt], dDiscountedSpot[opt], dStrike[opt], dYte[opt], dVolatility[opt], bIsCall[opt]);
			pBlackAndScholes[opt] = dVolatility[opt];//p;
			if(bIsAmerican[opt] && (!bIsCall[opt] || DC[opt] > 0))
				dVolatility[opt] = fmaxf((theoPriceCR1[opt]+theoPriceCR2[opt])/2.0, p) - dOptionPrice[opt];
			else 
				dVolatility[opt] = p - dOptionPrice[opt];
				
			//dVolatility[opt] = ((theoPriceCR1[opt]+theoPriceCR2[opt])/2.0) - dOptionPrice[opt];
		}
	}

}



#endif // #ifndef COXROSS


#ifndef _GPU_GREEKS_KERNEL_H_
#define _GPU_GREEKS_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <malloc.h>
#include <memory.h>
#include <OptionCalcGPU_sys.h>
/*
#define GPU_GREEKS_PER_OPTION 11

#define GPU_GREEK_THEO_PRICE					0
#define GPU_GREEK_ALPHA								1
#define GPU_GREEK_DELTA								2
#define GPU_GREEK_GAMMA								3
#define GPU_GREEK_VEGA								4
#define GPU_GREEK_THETA								5
#define GPU_GREEK_RHO									6
#define GPU_GREEK_DELTA_VEGA					7
#define GPU_GREEK_DELTA_THETA					8
#define GPU_GREEK_GAMMA_VEGA					9
#define GPU_GREEK_GAMMA_THETA					10
*/

static float* dGreeksCR1;
static float* dGreeksCR2;
static float* dGreeksCR;
static float* dGreeksCRdIV;
static float* dGreeksCRdRate;

static float* theoPriceCR3;
static float* theoPriceCR4;

static float* dVolatilityWithDeltaStep;
static float* dContRdWithDeltaStep;


static const __device__ int c00 = 0;
static const __device__ int c10 = 1;
static const __device__ int c11 = 2;
static const __device__ int c20 = 4;
static const __device__ int c21 = 5;
static const __device__ int c22 = 6;
static const __device__ int c31 = 7;
static const __device__ int c32 = 8;
static const __device__ int c41 = 9;
static const __device__ int c42 = 10;
static const __device__ int c43 = 11;
static const __device__ int coxRossGRArrayLen = 12;




/*
static const __device__ float cdGPUDeltaTime					= 1.0f / 365.0f;
static const __device__ float cdGPUDeltaVolatility		= 0.01f;
static const __device__ float cdGPUDeltaRate					= 0.01f;
*/

__device__ float NormalDensity_device(float X)
{
	return exp(-X*X / 2.) / sqrt(2. * GPU_Pi);
}


__device__ float CalculateGreeks_BlackAndScholes_device(	
																				float dContRd,
																				float dContRf,
																				float dDiscountedSpot,//dSpotPrice,
																				float dStrike,
																				float dYte,
																				float dVolatility,
																				int	bIsCall,
																				float* dGreeks)
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


	const float	dd1dS = 1.0f / (D * dDiscountedSpot);
	const float	dd1dT = -d2 / (2.0f * dYte) + (dContRd - dContRf) / D;
	const float	dd2dT = -d1 / (2.0f * dYte) + (dContRd - dContRf) / D;
	const float	dd1dVol = - d2 / dVolatility;
	const float	dd2dVol = - d1 / dVolatility;
	const float	dd12dR = dYte / D;
	const float n1 = NormalDensity_device(d1);
	const float n2 = NormalDensity_device(d2);


	//greeks
	dGreeks[GPU_GREEK_DELTA]				= exp(-dContRf * dYte) * (N1 - (bIsCall ? 0.0f : 1.0f));
	dGreeks[GPU_GREEK_GAMMA]				= exp(-dContRf * dYte) * dd1dS * n1;

	dGreeks[GPU_GREEK_THETA]				= ( A * (dContRf * (N1 - (bIsCall ? 0.0f : 1.0f)) - dd1dT * n1) -
		B * (dContRd * (N2 - (bIsCall ? 0.0f : 1.0f)) - dd2dT * n2) ) * cdGPUDeltaTime;            
	dGreeks[GPU_GREEK_DELTA_THETA]	= ( dContRf * (dGreeks[GPU_GREEK_DELTA] - (bIsCall ? 0.0f : exp(-dContRf * dYte) )) - 
		dGreeks[GPU_GREEK_GAMMA] * dd1dT / dd1dS ) * cdGPUDeltaTime;
	dGreeks[GPU_GREEK_GAMMA_THETA]	= ( dGreeks[GPU_GREEK_GAMMA] * (dContRf + 1.0f / (2.0f * dYte) + d1 * dd1dT) ) * cdGPUDeltaTime;
	dGreeks[GPU_GREEK_DELTA_VEGA]		= ( dGreeks[GPU_GREEK_GAMMA] * dd1dVol / dd1dS ) * cdGPUDeltaVolatility;
	dGreeks[GPU_GREEK_GAMMA_VEGA]		= ( -dGreeks[GPU_GREEK_GAMMA] * ( 1.0f / dVolatility + d1 * dd1dVol) ) * cdGPUDeltaVolatility;
	dGreeks[GPU_GREEK_VEGA]					= ( A * dd1dVol * n1 - B * dd2dVol * n2 ) * cdGPUDeltaVolatility;

	dGreeks[GPU_GREEK_RHO]					= ( (A * n1 - B * n2) * dd12dR + dYte * B * (N2 - (bIsCall ? 0.0f : 1.0f)) ) * cdGPUDeltaRate;
	
		// Check values
	dGreeks[GPU_GREEK_DELTA] = fmin(fmax(dGreeks[GPU_GREEK_DELTA], -1.0f), 1.0f);
	dGreeks[GPU_GREEK_GAMMA] = fmax(dGreeks[GPU_GREEK_GAMMA], 0.0f);
	dGreeks[GPU_GREEK_VEGA] = fmax(dGreeks[GPU_GREEK_VEGA], 0.0f);
	dGreeks[GPU_GREEK_THETA] = fmin(dGreeks[GPU_GREEK_THETA], 0.0f);


	// ALPHA
	if (ValueNEQZero_device(dGreeks[GPU_GREEK_THETA]))
	{
		dGreeks[GPU_GREEK_ALPHA]				= dGreeks[GPU_GREEK_GAMMA] / dGreeks[GPU_GREEK_THETA];
	}
	dGreeks[GPU_GREEK_THEO_PRICE]		= dTheoPrice;




	return  dTheoPrice;
}



__device__ float   
CalculateGreeksPut_coxRossGPU_device(
									const float* S,		// Discounted underlying spot price
									const float* K,		// Strike price
									const float Rd,	// Domestic continuos risk free rate
									float Rf,	// Foreign continuos risk free rate (or yield value)
									const float* V,		// Volatility
									const float  T,		// Years amount till expiration
									const int	Steps,// Amount of binomial tree steps		
									float* DP,
									//int DC, //Dividends payments
									int opt,
									float* dGreeks
									)

{
	long grkIdx = 0;

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
	__shared__	float C[coxRossGRArrayLen];
	

	for(int i = threadIdx.x; i < coxRossGRArrayLen; i += blockDim.x) {
		C[i] = badGPUFloatValue;
	}

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
		const float _S	= S0 * (expc[i] = expf((-U) * (Steps - i)));
		const float C	= (_K - _S);

		TreeA[i] = _Tree[i] = fmaxf(C, 0.0f);
	}

	for(int k = 1; k <= Steps ;){


		//koefficients for greeks:
		long greekTreeStep = Steps - k + 1;
		if(greekTreeStep == 4)
		{
				C[c41] = TreeA[k + 1];
				C[c42] = TreeA[k + 3];
				C[c43] = TreeA[k + 5];
		}
		else if(greekTreeStep == 3)
		{
				C[c31] = TreeA[k + 1];
				C[c32] = TreeA[k + 3];
		}
		else if(greekTreeStep == 2)
		{
				C[c20] = TreeA[k - 1];
				C[c21] = TreeA[k + 1];
				C[c22] = TreeA[k + 3];
		}
		else if(greekTreeStep == 1)
		{
				C[c10] = TreeA[k - 1];
				C[c11] = TreeA[k + 1]; 
		}

		__syncthreads();

		for(int l = threadIdx.x; l <= Steps - k; l += blockDim.x) {
			
			float L = TreeA[sh(l)-1];
			float R = TreeA[sh(l)+1];
			
			if(dpv[Steps - k + 1]>FLT_EPSILON && DP!=NULL) {
				const float _S =S0 + dpv[Steps - k + 1];
				const float S1 = _S * expc[sh(l)-1];
				const float S2 = _S * expc[sh(l)+1];
				const float S3 = _S * expc[sh(l)];
				L = fmaxf(_K - S1, L);
				R = fmaxf(_Tree[sh(l)+1] = fmaxf(_K - S2, 0.0), R);
				_Tree[sh(l)] =  fmaxf(_K - S3, 0.0);
				
			}
			TreeA[sh(l)] = fmaxf(Pu * R + Pd * L, _Tree[sh(l)]);
				

		}
		k++;

	}


	__syncthreads();

	float dTheoPrice = TreeA[Steps]; // return option value from top of the tree

	if(!IsValidTheoPrice_device(dTheoPrice))
	{
		dTheoPrice = GPUBadTheoPrice;
	}

	
	//calculate greeks
	dGreeks[grkIdx + GPU_GREEK_THEO_PRICE] = C[c00] = dTheoPrice;


	const float Tau = T / Steps;
	const float u = exp(U);
	const float d = exp(D);
	const float u_d = u - d;


	
	//DELTA
	if ( ValueNEQZero_device(u_d) && ValueNEQZero_device(S0) && !IsBadValue_device(C[c11]) && !IsBadValue_device(C[c10]))
	{
		dGreeks[grkIdx + GPU_GREEK_DELTA] = (C[c11] - C[c10]) / (S0 * u_d);
		dGreeks[grkIdx + GPU_GREEK_DELTA] = fmin(fmax(dGreeks[grkIdx + GPU_GREEK_DELTA], -1.), 1.);
	}


  // GAMMA
	if (ValueNEQZero_device(u_d) && ValueNEQZero_device(S0) && ValueNEQZero_device(u) && ValueNEQZero_device(d) && !IsBadValue_device(C[c20]) && !IsBadValue_device(C[c21]) && !IsBadValue_device(C[c22]))
	{
		float dDeltaU = (C[c22] - C[c21]) / (S0 * u_d * u);
		float dDeltaD = (C[c21] - C[c20]) / (S0 * u_d * d);

		dGreeks[grkIdx + GPU_GREEK_GAMMA] = fmax((dDeltaU - dDeltaD) / (S0 * u_d), 0.0f);
	}



	// THETA
	if (ValueNEQZero_device(Tau) && !IsBadValue_device(C[c21]) && !IsBadValue_device(C[c00]))
	{
		dGreeks[grkIdx + GPU_GREEK_THETA] = (C[c21] - C[c00]) / (730.0 * Tau);
	}

	// DELTA THETA
	if (ValueNEQZero_device(u_d) && ValueNEQZero_device(S0) && ValueNEQZero_device(u) && ValueNEQZero_device(d) && ValueNEQZero_device(Tau) && !IsBadValue_device(C[c32]) && !IsBadValue_device(C[c31]))
	{
		float dDeltaM = (C[c32] - C[c31]) / (S0 * u_d * u * d);
		dGreeks[grkIdx + GPU_GREEK_DELTA_THETA] = (dDeltaM - dGreeks[grkIdx + GPU_GREEK_DELTA]) / (730.0 * Tau);
	}

	// GAMMA THETA
	if (ValueNEQZero_device(u_d) && ValueNEQZero_device(S0) && ValueNEQZero_device(u) && ValueNEQZero_device(d) && ValueNEQZero_device(Tau) && !IsBadValue_device(C[c41]) && !IsBadValue_device(C[c42]) && !IsBadValue_device(C[c43]) )
	{
		float dDeltaUU = (C[c43] - C[c42]) / (S0 * u_d * u * u * d);
		float dDeltaDD = (C[c42] - C[c41]) / (S0 * u_d * u * d * d);
		float dGammaM = (dDeltaUU - dDeltaDD) / (S0 * u_d * u * d );

		dGreeks[grkIdx + GPU_GREEK_GAMMA_THETA] = fmin((dGammaM - dGreeks[grkIdx + GPU_GREEK_GAMMA]) / (730.0 * Tau), 0.0f);
	}

		// Check values
	dGreeks[grkIdx + GPU_GREEK_DELTA] = fmin(fmax(dGreeks[grkIdx + GPU_GREEK_DELTA], -1.0f), 1.0f);
	dGreeks[grkIdx + GPU_GREEK_GAMMA] = fmax(dGreeks[grkIdx + GPU_GREEK_GAMMA], 0.0f);
	dGreeks[grkIdx + GPU_GREEK_THETA] = fmin(dGreeks[grkIdx + GPU_GREEK_THETA], 0.0f);

	// ALPHA
	if ((ValueNEQZero_device(dGreeks[grkIdx + GPU_GREEK_THETA])))
	{
		dGreeks[grkIdx + GPU_GREEK_ALPHA] = dGreeks[grkIdx + GPU_GREEK_GAMMA] / dGreeks[grkIdx + GPU_GREEK_THETA];
	}
	//dGreeks[grkIdx + GPU_GREEK_ALPHA] = ValueNEQZero_device(dGreeks[grkIdx + GPU_GREEK_THETA]);

	return dTheoPrice;
	//return grkIdx;

}

__device__ float   
CalculateGreeksCall_coxRossGPU_device(
									const float* S,		// Discounted underlying spot price
									const float* K,		// Strike price
									const float Rd,	// Domestic continuos risk free rate
									float Rf,	// Foreign continuos risk free rate (or yield value)
									const float* V,		// Volatility
									const float  T,		// Years amount till expiration
									const int	Steps,// Amount of binomial tree steps		
									float* DP,
									//int DC, //Dividends payments
									int opt,
									float* dGreeks
									)
{
	long grkIdx = 0;

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
	__shared__	float C[coxRossGRArrayLen];
	

	for(int i = threadIdx.x; i < coxRossGRArrayLen; i += blockDim.x) {
		C[i] = badGPUFloatValue;
	}

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
		const float _S	= S0 * (expc[i] = expf((U) * (Steps - i)));
		const float C	= (_S - _K);

		TreeA[i] = _Tree[i] = fmaxf(C, 0.0f);
	}

	for(int k = 1; k <= Steps ;){


		//koefficients for greeks:
		long greekTreeStep = Steps - k + 1;
		if(greekTreeStep == 4)
		{
				C[c41] = TreeA[k + 5];
				C[c42] = TreeA[k + 3];
				C[c43] = TreeA[k + 1];

		}
		else if(greekTreeStep == 3)
		{
				C[c31] = TreeA[k + 3];
				C[c32] = TreeA[k + 1];
		}
		else if(greekTreeStep == 2)
		{
				C[c20] = TreeA[k + 3];
				C[c21] = TreeA[k + 1];
				C[c22] = TreeA[k - 1];
		}
		else if(greekTreeStep == 1)
		{
				C[c10] = TreeA[k + 1];
				C[c11] = TreeA[k - 1];
		}

		__syncthreads();

		for(int l = threadIdx.x; l <= Steps - k; l += blockDim.x) {
			
			float L = TreeA[sh(l)-1];
			float R = TreeA[sh(l)+1];
			
			if(dpv[Steps - k + 1]>FLT_EPSILON && DP!=NULL) {
				const float _S =S0 + dpv[Steps - k + 1];
				const float S1 = _S * expc[sh(l)-1];
				const float S2 = _S * expc[sh(l)+1];
				const float S3 = _S * expc[sh(l)];
				L = fmaxf(S1 - _K, L);
				R = fmaxf(_Tree[sh(l)+1] = fmaxf(S2 - _K, 0.0), R);
				_Tree[sh(l)] =  fmaxf(S3 - _K, 0.0);
				
			}
			TreeA[sh(l)] = fmaxf(Pu * L + Pd * R, _Tree[sh(l)]);
				
		}
		k++;

	}


	__syncthreads();

	float dTheoPrice = TreeA[Steps]; // return option value from top of the tree

	if(!IsValidTheoPrice_device(dTheoPrice))
	{
		dTheoPrice = GPUBadTheoPrice;
	}

	
	//calculate greeks
	dGreeks[grkIdx + GPU_GREEK_THEO_PRICE] = C[c00] = dTheoPrice;

	const float Tau = T / Steps;
	const float u = exp(U);
	const float d = exp(D);
	const float u_d = u - d;

	
	//DELTA
	if ( ValueNEQZero_device(u_d) && ValueNEQZero_device(S0) && !IsBadValue_device(C[c11]) && !IsBadValue_device(C[c10]))
	{
		dGreeks[grkIdx + GPU_GREEK_DELTA] = (C[c11] - C[c10]) / (S0 * u_d);
		dGreeks[grkIdx + GPU_GREEK_DELTA] = fmin(fmax(dGreeks[grkIdx + GPU_GREEK_DELTA], -1.), 1.);
	}


  // GAMMA
	if (ValueNEQZero_device(u_d) && ValueNEQZero_device(S0) && ValueNEQZero_device(u) && ValueNEQZero_device(d) && !IsBadValue_device(C[c20]) && !IsBadValue_device(C[c21]) && !IsBadValue_device(C[c22]))
	{
		float dDeltaU = (C[c22] - C[c21]) / (S0 * u_d * u);
		float dDeltaD = (C[c21] - C[c20]) / (S0 * u_d * d);

		dGreeks[grkIdx + GPU_GREEK_GAMMA] = fmax((dDeltaU - dDeltaD) / (S0 * u_d), 0.0f);
	}



	// THETA
	if (ValueNEQZero_device(Tau) && !IsBadValue_device(C[c21]) && !IsBadValue_device(C[c00]))
	{
		dGreeks[grkIdx + GPU_GREEK_THETA] = (C[c21] - C[c00]) / (730.0 * Tau);
	}

	// DELTA THETA
	if (ValueNEQZero_device(u_d) && ValueNEQZero_device(S0) && ValueNEQZero_device(u) && ValueNEQZero_device(d) && ValueNEQZero_device(Tau) && !IsBadValue_device(C[c32]) && !IsBadValue_device(C[c31]))
	{
		float dDeltaM = (C[c32] - C[c31]) / (S0 * u_d * u * d);
		dGreeks[grkIdx + GPU_GREEK_DELTA_THETA] = (dDeltaM - dGreeks[grkIdx + GPU_GREEK_DELTA]) / (730.0 * Tau);
	}

	// GAMMA THETA
	if (ValueNEQZero_device(u_d) && ValueNEQZero_device(S0) && ValueNEQZero_device(u) && ValueNEQZero_device(d) && ValueNEQZero_device(Tau) && !IsBadValue_device(C[c41]) && !IsBadValue_device(C[c42]) && !IsBadValue_device(C[c43]) )
	{
		float dDeltaUU = (C[c43] - C[c42]) / (S0 * u_d * u * u * d);
		float dDeltaDD = (C[c42] - C[c41]) / (S0 * u_d * u * d * d);
		float dGammaM = (dDeltaUU - dDeltaDD) / (S0 * u_d * u * d );

		dGreeks[grkIdx + GPU_GREEK_GAMMA_THETA] = fmin((dGammaM - dGreeks[grkIdx + GPU_GREEK_GAMMA]) / (730.0 * Tau), 0.0f);
	}

		// Check values
	dGreeks[grkIdx + GPU_GREEK_DELTA] = fmin(fmax(dGreeks[grkIdx + GPU_GREEK_DELTA], -1.0f), 1.0f);
	dGreeks[grkIdx + GPU_GREEK_GAMMA] = fmax(dGreeks[grkIdx + GPU_GREEK_GAMMA], 0.0f);
	dGreeks[grkIdx + GPU_GREEK_THETA] = fmin(dGreeks[grkIdx + GPU_GREEK_THETA], 0.0f);

	// ALPHA
	if ((ValueNEQZero_device(dGreeks[grkIdx + GPU_GREEK_THETA])))
	{
		dGreeks[grkIdx + GPU_GREEK_ALPHA] = dGreeks[grkIdx + GPU_GREEK_GAMMA] / dGreeks[grkIdx + GPU_GREEK_THETA];
	}
	//dGreeks[grkIdx + GPU_GREEK_ALPHA] = ValueNEQZero_device(dGreeks[grkIdx + GPU_GREEK_THETA]);


	return dTheoPrice;
	//return grkIdx;

}

// Calculate CR

__global__ void CalculateGreeks_gpuCR(
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
					 int optCount,
					 int* dCalcFlag,
					 float* theoPrice,//out,
					 float* dGreeks
					 )
{
	/*
	theoPrice[0] = 0.15;
	theoPrice[1] = 0.2;


	dGreeks[0] = 0.5;
	dGreeks[1] = 0.4;

	return;
	*/

	for(int opt = blockIdx.x; opt < optCount; opt += gridDim.x) {
		if(IsAmerican[opt] && (!IsCall[opt] || DC[opt] > 0) && (V[opt] > 0) && (dCalcFlag[opt]>-1) )
		{
			theoPrice[opt] = IsCall[opt] ? CalculateGreeksCall_coxRossGPU_device(
				S,		// Underlying spot price
				K,		// Strike price
				dContRd[opt],		// Domestic continuos risk free rate
				dContRf[opt],		// Foreign continuos risk free rate (or yield value)
				V,		// Volatility
				dYte[opt],		// Years amount till expiration
				Steps,
				DC[opt]>0?DP+opt*STEPS_LENGTH:NULL,
			opt,//out/
			dGreeks + opt*GPU_GREEKS_PER_OPTION
			):
			CalculateGreeksPut_coxRossGPU_device(
				S,		// Underlying spot price
				K,		// Strike price
				dContRd[opt],		// Domestic continuos risk free rate
				dContRf[opt],		// Foreign continuos risk free rate (or yield value)
				V,		// Volatility
				dYte[opt],		// Years amount till expiration
				Steps,
				DC[opt]>0?DP+opt*STEPS_LENGTH:NULL,
				opt,//out/
				dGreeks + opt*GPU_GREEKS_PER_OPTION);

		}
	}
}


__global__  void	AdjustCR(float* theoPriceCR1, 
																float* theoPriceCR2, 
																float* dGreeksCR1, 
																float* dGreeksCR2,  
																int		optCount)
{
	const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	for(int opt = tid; opt < optCount; opt += THREAD_N) 
	{
		theoPriceCR1[opt]		= (theoPriceCR1[opt] + theoPriceCR2[opt])/2.0f;

		int idx = opt*GPU_GREEKS_PER_OPTION;
		dGreeksCR1[idx + GPU_GREEK_THEO_PRICE]		= (dGreeksCR1[idx + GPU_GREEK_THEO_PRICE]			+ dGreeksCR2[idx + GPU_GREEK_THEO_PRICE])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_ALPHA]					= (dGreeksCR1[idx + GPU_GREEK_ALPHA]					+ dGreeksCR2[idx + GPU_GREEK_ALPHA])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_DELTA]					= (dGreeksCR1[idx + GPU_GREEK_DELTA]					+ dGreeksCR2[idx + GPU_GREEK_DELTA])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_GAMMA]					= (dGreeksCR1[idx + GPU_GREEK_GAMMA]					+ dGreeksCR2[idx + GPU_GREEK_GAMMA])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_VEGA]					= (dGreeksCR1[idx + GPU_GREEK_VEGA]						+ dGreeksCR2[idx + GPU_GREEK_VEGA])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_THETA]					= (dGreeksCR1[idx + GPU_GREEK_THETA]					+ dGreeksCR2[idx + GPU_GREEK_THETA])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_RHO]						= (dGreeksCR1[idx + GPU_GREEK_RHO]						+ dGreeksCR2[idx + GPU_GREEK_RHO])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_DELTA_VEGA]		= (dGreeksCR1[idx + GPU_GREEK_DELTA_VEGA]			+ dGreeksCR2[idx + GPU_GREEK_DELTA_VEGA])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_DELTA_THETA]		= (dGreeksCR1[idx + GPU_GREEK_DELTA_THETA]		+ dGreeksCR2[idx + GPU_GREEK_DELTA_THETA])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_GAMMA_VEGA]		= (dGreeksCR1[idx + GPU_GREEK_GAMMA_VEGA]			+ dGreeksCR2[idx + GPU_GREEK_GAMMA_VEGA])/2.0f;
		dGreeksCR1[idx + GPU_GREEK_GAMMA_THETA]		= (dGreeksCR1[idx + GPU_GREEK_GAMMA_THETA]		+ dGreeksCR2[idx + GPU_GREEK_GAMMA_THETA])/2.0f;
	}
}

//adjust cox ross with BS (First iteration)
__global__  void	AdjustCRBS_FI(
																					float* dContRd,
																					float* dContRf,
																					float* dDiscountedSpot,//Discounted spot price,
																					float* dStrike,
																					float* dYte,
																					float* dVolatility,
																					int*	bIsCall,
																					int*	bIsAmerican,
																					int* DC,
																					int* dCalcFlag,
																					float* theoPriceCR, 
																					float* dGreeks, 
																					float* dGreeksCR, 
																					int		optCount)
{
	const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	for(int opt = tid; opt < optCount; opt += THREAD_N) 
	{

		float theoPriceBS = CalculateGreeks_BlackAndScholes_device(
			dContRd[opt], 
			DC[opt]>0?0:dContRf[opt], 
			dDiscountedSpot[opt], 
			dStrike[opt], 
			dYte[opt], 
			dVolatility[opt], 
			bIsCall[opt], 
			dGreeks + opt*GPU_GREEKS_PER_OPTION);

		if (theoPriceCR[opt] > theoPriceBS)
		{
			int idx = opt*GPU_GREEKS_PER_OPTION;

			dGreeks[idx + GPU_GREEK_THEO_PRICE]			= dGreeksCR[idx + GPU_GREEK_THEO_PRICE];
			dGreeks[idx + GPU_GREEK_ALPHA]					= dGreeksCR[idx + GPU_GREEK_ALPHA];
			dGreeks[idx + GPU_GREEK_DELTA]					= dGreeksCR[idx + GPU_GREEK_DELTA];
			dGreeks[idx + GPU_GREEK_GAMMA]					= dGreeksCR[idx + GPU_GREEK_GAMMA];
			dGreeks[idx + GPU_GREEK_VEGA]						= dGreeksCR[idx + GPU_GREEK_VEGA];
			dGreeks[idx + GPU_GREEK_THETA]					= dGreeksCR[idx + GPU_GREEK_THETA];
			dGreeks[idx + GPU_GREEK_RHO]						= dGreeksCR[idx + GPU_GREEK_RHO];
			dGreeks[idx + GPU_GREEK_DELTA_VEGA]			= dGreeksCR[idx + GPU_GREEK_DELTA_VEGA];
			dGreeks[idx + GPU_GREEK_DELTA_THETA]		= dGreeksCR[idx + GPU_GREEK_DELTA_THETA];
			dGreeks[idx + GPU_GREEK_GAMMA_VEGA]			= dGreeksCR[idx + GPU_GREEK_GAMMA_VEGA];
			dGreeks[idx + GPU_GREEK_GAMMA_THETA]		= dGreeksCR[idx + GPU_GREEK_GAMMA_THETA];
		
		}else{
			dCalcFlag[opt]		= -1;
		}

	}
}


//adjust cox ross with BS (Second iteration)
__global__  void	AdjustCRBS_SI(
																					float* dContRd,
																					float* dContRf,
																					float* dDiscountedSpot,//Discounted spot price,
																					float* dStrike,
																					float* dYte,
																					float* dVolatility,
																					int*	bIsCall,
																					int*	bIsAmerican,
																					int* DC,
																					int* dCalcFlag,
																					float* theoPriceCRAdj, 
																					float* dGreeks, 
																					float* dGreeksCRAdj, 
																					float* dGreeksCR, 
																					int		optCount)
{
	const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	for(int opt = tid; opt < optCount; opt += THREAD_N) 
	{
		if (dCalcFlag[opt] == -1){
			continue;
		}
		
		
		float theoPriceBS = CalculateGreeks_BlackAndScholes_device(
			dContRd[opt], 
			DC[opt]>0?0:dContRf[opt], 
			dDiscountedSpot[opt], 
			dStrike[opt], 
			dYte[opt], 
			dVolatility[opt], 
			bIsCall[opt], 
			dGreeks + opt*GPU_GREEKS_PER_OPTION);
		
	

		int idx = opt*GPU_GREEKS_PER_OPTION;
		
		float* gr = theoPriceBS > theoPriceCRAdj[opt] ? dGreeks: dGreeksCRAdj;


		dGreeks[idx + GPU_GREEK_VEGA]				= fmax(gr[idx + GPU_GREEK_THEO_PRICE] - dGreeksCR[idx + GPU_GREEK_THEO_PRICE], 0.0f);
		dGreeks[idx + GPU_GREEK_DELTA_VEGA] = gr[idx + GPU_GREEK_DELTA] - dGreeksCR[idx + GPU_GREEK_DELTA];
		dGreeks[idx + GPU_GREEK_GAMMA_VEGA] = gr[idx + GPU_GREEK_GAMMA] - dGreeksCR[idx + GPU_GREEK_GAMMA];

		dGreeks[idx + GPU_GREEK_THEO_PRICE]			= dGreeksCR[idx + GPU_GREEK_THEO_PRICE];
		dGreeks[idx + GPU_GREEK_ALPHA]					= dGreeksCR[idx + GPU_GREEK_ALPHA];
		dGreeks[idx + GPU_GREEK_DELTA]					= dGreeksCR[idx + GPU_GREEK_DELTA];
		dGreeks[idx + GPU_GREEK_GAMMA]					= dGreeksCR[idx + GPU_GREEK_GAMMA];
		//dGreeks[idx + GPU_GREEK_VEGA]						= dGreeksCR[idx + GPU_GREEK_VEGA];
		dGreeks[idx + GPU_GREEK_THETA]					= dGreeksCR[idx + GPU_GREEK_THETA];
		dGreeks[idx + GPU_GREEK_RHO]						= dGreeksCR[idx + GPU_GREEK_RHO];

	}
}

//adjust cox ross with BS (Third iteration)
__global__  void	AdjustCRBS_TI(
																					float* dContRd,
																					float* dContRf,
																					float* dDiscountedSpot,//Discounted spot price,
																					float* dStrike,
																					float* dYte,
																					float* dVolatility,
																					int*	bIsCall,
																					int*	bIsAmerican,
																					int* DC,
																					int* dCalcFlag,
																					float* theoPriceCRAdj, 
																					float* dGreeks, 
																					float* dGreeksCRAdj, 
																					float* dGreeksCR, 
																					int		optCount)
{
	const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	for(int opt = tid; opt < optCount; opt += THREAD_N) 
	{
		
		if (dCalcFlag[opt] == -1){
			continue;
		}
		
		
		float theoPriceBS = CalculateGreeks_BlackAndScholes_device(
			dContRd[opt], 
			DC[opt]>0?0:dContRf[opt], 
			dDiscountedSpot[opt], 
			dStrike[opt], 
			dYte[opt], 
			dVolatility[opt], 
			bIsCall[opt], 
			dGreeks + opt*GPU_GREEKS_PER_OPTION);
		
	

		int idx = opt*GPU_GREEKS_PER_OPTION;

		float theoPrice = theoPriceBS > dGreeksCRAdj[idx + GPU_GREEK_THEO_PRICE] ? theoPriceBS: dGreeksCRAdj[idx + GPU_GREEK_THEO_PRICE];

		dGreeksCR[idx + GPU_GREEK_RHO]				= theoPrice - dGreeksCR[idx + GPU_GREEK_THEO_PRICE];
		//dGreeksCR[idx + GPU_GREEK_RHO]				= 0.7;
		//dGreeks[idx + GPU_GREEK_RHO]				= 0.7;
		//dGreeksCRAdj[idx + GPU_GREEK_RHO]				= 0.7;

	}
}

__global__ void CalculateGreeks_gpuCREABS(					
				float*	dGreeks,
				float*	dGreeksCRdVola,
				float*	dGreeksCRdRate,
				int			optCount
	) 
{

		const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
		const int THREAD_N = blockDim.x * gridDim.x;

		for(int opt = tid; opt < optCount; opt += THREAD_N) 
		{			
			int idx = opt*GPU_GREEKS_PER_OPTION;

			dGreeks[idx + GPU_GREEK_VEGA]					= fmaxf(dGreeks[idx + GPU_GREEK_THEO_PRICE] - dGreeksCRdVola[idx + GPU_GREEK_THEO_PRICE], 0.0f);
			dGreeks[idx + GPU_GREEK_DELTA_VEGA]		= dGreeks[idx + GPU_GREEK_DELTA] - dGreeksCRdVola[idx + GPU_GREEK_DELTA];
			dGreeks[idx + GPU_GREEK_GAMMA_VEGA]		= dGreeks[idx + GPU_GREEK_GAMMA] - dGreeksCRdVola[idx + GPU_GREEK_GAMMA];
			dGreeks[idx + GPU_GREEK_RHO]					= dGreeksCRdRate[idx + GPU_GREEK_RHO];

		}


}


#endif // #ifndef _GPU_GREEKS_KERNEL_H_

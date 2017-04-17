
#pragma once

//#include <windows.h>
#include <math.h>
#include <float.h>
#include <malloc.h>
#include <memory.h>

#include "Logger.h"

//#include <string>
//#include <vector>


//#include <stdlib.h>
//#include <xutility>
//#include <algorithm>

const long		DEFAULT_BINOMIAL_STEPS = 100;

#define  MM_EGAR_BS			0x001
#define  MM_EGAR_BINOMIAL	0x002
#define  MM_EGAR_OPTIMIZED	0x003
#define  MM_EGAR_VSKLOG		0x004

#include <limits>
//=============================================================================
// Pi constant value
//=============================================================================
const float Pi = 3.1415926535897932384626433832795f;

//=============================================================================
// Epsilon constant value
//=============================================================================
const float Epsilon = 0.000001f;

//=============================================================================
// Bad float value
//=============================================================================
const float BadDoubleValue = -3.4E+38f;
//=============================================================================
// Max theo price value
//=============================================================================
const float MaxTheoPriceValue = 10000.;

bool IsBadValue(float dVal);

extern "C" 
bool ValueNEQZero(float dVal);

bool IsValidTheoPrice(float& dTheoPrice);


const float cdDaysPerYear360 = 360.0f;

const float cdDaysPerYear365 = 365.0f;

const float cdDeltaVolatility = 0.01f;

const float cdDeltaRate = 0.01f;

const float cdDeltaTime = 1.0f / 365.0f;

const float cdDeltaSqrtTime = 0.0523f;

enum GREEK_TYPE
{
	GT_NOTHING	 = 0x00000000,

	GT_THEOPRICE	= 0x00000001,
	GT_ALPHA		= 0x00000002,
	GT_DELTA		= 0x00000004,
	GT_GAMMA		= 0x00000008,
	GT_VEGA			= 0x00000010,
	GT_THETA		= 0x00000020,
	GT_RHO			= 0x00000040,
	GT_RHO2			= 0x00000080,

	GT_DELTA_VEGA	= 0x00000100,
	GT_DELTA_THETA	= 0x00000200,
	GT_GAMMA_VEGA	= 0x00000400,
	GT_GAMMA_THETA	= 0x00000800,

	GT_ALL			= 0xFFFFFFFF
};

enum FREQUENCY
{
	FREQUENCY_MONTHLY	  = 12,
	FREQUENCY_QUATERLY	  = 4,
	FREQUENCY_SEMIANNUALY = 2,
	FREQUENCY_ANNUALY	  = 1
};

enum VOLATILITY_FLAG
{
	VF_OK	        = 0x00000000,
	VF_TOOLOW       = 0x00000001,
	VF_TOOHIGH      = 0x00000002,
	VF_UNRELIABLE	= 0x00000003
};


struct GREEKS
{
	long	nMask;

	float	dTheoPrice;
	float  dAlpha;
	float	dDelta;
	float	dGamma;
	float	dVega;
	float	dTheta;
	float	dRho;
	float	dRho2;

	float	dDeltaVega;
	float	dDeltaTheta;
	float	dGammaVega;
	float	dGammaTheta;

	GREEKS() { memset(this, '\0', sizeof(GREEKS)); }
};

//=============================================================================
// Max. and min. steps of binomial tree 
//=============================================================================
const long cnTreeStepsMax = 303;
const long cnTreeStepsMin = 5;

extern "C" 
float  CalcVolatilityMM3(float	dDomesticRate,	
										float	dForeignRate, 
										float	dSpotPrice,
										float	dOptionPrice,
										float	dStrike,
										long	nDTE,
										long	nIsCall,
										long	nIsAmerican,
										long	nCount, 
										float*	pDivAmnts,
										float*	pDivYears,
										long	nSteps,
										float	dSkew,
										float	dKurtosis,
										long	nModel,
										 long*   pnFlag,
										float  tol = 1.0e-6,
										float tolF = 1.0e-4);

extern "C" 
float CalculateOptionMM(float	dDomesticRate,
								float	dForeignRate, 
								float	dSpotPrice,
								float	dStrike,
								float	dVolatility, 
								long	nDTE,
								bool	bIsCall,
								bool	bIsAmerican,
								long	nCount, 
								float*	pDivAmnts,
								float*	pDivYears,
								long	nSteps,
								float	dSkew,
								float	dKurtosis,
								long	nModel,
								GREEKS*	pGreeks = NULL);

float CO_BlackScholes(	float	R,
						float	RF,
						float	S,
						float	K,
						float	V,
						int		Dte, 
                        bool	Call,
						bool	American,
						float* DivAmnt,
						float* DivYte,
						int		DivCount,
						GREEKS *pGreeks = NULL
                        );

float CO_StandardBinomial(
						float	R,
					    float	RF,
					    float	S,
					    float	K,
					    float	V,
					    int		Dte, 
                        bool	Call,
					    bool	American,						 
						float* DivAmnt,						   
						float* DivYte,						    
						int		DivCount,						    
						long	Steps,							
						GREEKS *pGreeks = NULL);

float BlackAndScholes(	float	dRateDomestic,
						float	dRateForeign,
						float	dSpotPrice,
						float	dStrike,
						int		nDte,
						float	dVolatility,
						bool	bIsCall,
						float* pdDivAmnt,						   
						float* pdDivYte,						    
						int		nDivCount,
						GREEKS*	pGreeks = NULL/*out*/
						);

float CO_CoxRossWithBlackScholes(	
						float	R,
						float	RF,
						float	S,
						float	K,
						float	V,
						int		Dte, 
                        bool	Call,
						bool	American,
						float* DivAmnt,
						float* DivYte,
						int		DivCount,
						long	Steps,
						GREEKS *pGreeks = NULL
                        );

float CoxRossOddEvenAdjust(
				float	dSpotPrice,
				float	dStrike,
				float	dRateDomestic,
				float	dRateForeign,
				float	dVolatility,
				int		nDte,
				bool	bIsCall,
				long	nSteps,	// Amount of binomial tree steps
				float* pdDivAmnt,						   
				float* pdDivYte,						    
				int		nDivCount,
				GREEKS*	pGreeks = NULL /*out*/
				);

extern "C" 
float CoxRoss(	float	S,		// Underlying spot price
				float	K,		// Strike price
				float	Rd,		// Domestic continuos risk free rate
				float	Rf,		// Foreign continuos risk free rate (or yield value)
				float	V,		// Volatility
				float	T,		// Years amount till expiration
				bool	IsCall,	// true if it's 'Call option', false is 'Put option'
				long	Steps,	// Amount of binomial tree steps		
				float*	DA,		// Array of dividend's amount
				float*	DT,		// Array of years till dividend payment
				long	DC,		// Count of dividends
				GREEKS*	pGreeks//out/
				);

extern "C" 
float CoxRossMT(	float	S,		// Underlying spot price
				float	K,		// Strike price
				float	Rd,		// Domestic continuos risk free rate
				float	Rf,		// Foreign continuos risk free rate (or yield value)
				float	V,		// Volatility
				float	T,		// Years amount till expiration
				bool	IsCall,	// true if it's 'Call option', false is 'Put option'
				long	Steps,	// Amount of binomial tree steps		
				float*	DA,		// Array of dividend's amount
				float*	DT,		// Array of years till dividend payment
				long	DC,		// Count of dividends
				GREEKS*	pGreeks//out/
				);

extern "C" 
long CalcGreeksMM2(	float	dDomesticRate,
									 float	dForeignRate, 
									 float	dSpotPrice,
									 float	dStrike,
									 float	dVolatility, 
									 long	nDTE,
									 long	nIsCall,
									 long	nIsAmerican,
									 long	nCount, 
									 float*	pDivAmnts,
									 float*	pDivYears,
									 long	nSteps,
									 float	dSkew,
									 float	dKurtosis,
									 long	nModel,
									 GREEKS*	pGreeks);


extern "C" 
long  GetDividendsCount( long nToday, long nDTE, 
									   long nLastDivDate, long nFrequency ); 

extern "C" 
long GetDividends2( long nToday, long nDTE, 
								  long nLastDivDate, long nFrequency, float dAmount, 
								  long nCount, float* pDivAmnts, float* pDivDays,
								  long *pnCount );

extern "C" 
float DividendPv(float dDivAmnt, float dCntRate, float dTime);

float NormalC(float X);

float NormalDensity(float X);

extern "C" 
float StringToDate(const char* str);
/************************************************************
	Discount asset price for the present value of 
	dividends.

	S			- Stock price
	R			- Domestic continuous rate
	pDivAmnts	- Array of dividends 
	pDivYte		- Array of time(amount of years) to payments
	nDivCount	- Size of dividends array
	T			- Time horizont for discount

	Return value - Discounted asset price
************************************************************/
float DiscountForDividends(float		S, 
							float		R,
							float*		pDivAmnt,
							float*		pDivYte,
							long			nDivCount,
							float		T);


extern "C" 
float RateDiscToCont(float dDiscRate, unsigned int nDays);



extern "C" 
bool ParseData(
							 const char*stocksf,                
							 const char* optionsf,	
							 const char* interestRatef,
							 const long todayDate,							
							 long**			optionID,								
							 long**			stockID,								
							 float**		stockPrice,							
							 float**		divAmounts,								
							 float**		divYears,							
							 float**		divYield,								
							 long**			divFreq,								
							 int**			isAmerican,							
							 int**			isCall,									
							 float**		strike,									
							 int**			DTE,										
							 float**		optionPrice,						
							 int**			divIndex,
							 int**			divCount,
							 float**		interestRate,
							 long*			nCount,
							 long*			nDivCount);								


extern "C" 
bool ParseDataVS(const char* objectInitRef,
							 const char* userLogin,
							 const char* userPassword,
							 const long todayDate,
							 long**			optionID,								
							 long**			stockID,								
							 float**		stockPrice,							
							 float**		divAmounts,								
							 float**		divYears,							
							 float**		divYield,								
							 long**			divFreq,								
							 int**			isAmerican,							
							 int**			isCall,									
							 float**		strike,									
							 int**			DTE,										
							 float**		optionPrice,						
							 int**			divIndex,
							 int**			divCount,
							 float**		interestRate,
							 long*			nCount,
							 long*			nDivCount);								

extern "C" 
bool ParseDataVSSSL(const char* objectInitRef,
							 const char* userLogin,
							 const char* userPassword,
							 const char* rootCertFilename,
							 const char* clientKeyFilename,
							 const char* clientKeyPassword,
							 const long todayDate,
							 long**			optionID,								
							 long**			stockID,								
							 float**		stockPrice,							
							 float**		divAmounts,								
							 float**		divYears,							
							 float**		divYield,								
							 long**			divFreq,								
							 int**			isAmerican,							
							 int**			isCall,									
							 float**		strike,									
							 int**			DTE,										
							 float**		optionPrice,						
							 int**			divIndex,
							 int**			divCount,
							 float**		interestRate,
							 long*			nCount,
							 long*			nDivCount);								

/*
extern "C" 
bool Pull(long**		changedOptionIdx,
					float**	  optionPrice,
					long*		  optionsCount,
					long**		changedStockIdx,
					float**	  stockPrice,
					long**		stockFstOptIdx,
					long**		stockLstOptIdx,
					long*		  stocksCount);
*/

extern "C" 
bool Pull(long**		changedOptionIdx,

					float**	  optionBidPrice,
					float**	  optionAskPrice,
					float**	  optionLastPrice,

					long**	  optionBidSize,
					long**	  optionAskSize,
					long**	  optionLastSize,

					double**	 optionBidDate,
					double**	 optionAskDate,
					double**	 optionLastDate,

					long**	  optionVolume,
					long*		  optionsCount,

					long**		changedStockIdx,

					float**	  stockBidPrice,
					float**	  stockAskPrice,
					float**	  stockLastPrice,

					long**	  stockBidSize,
					long**	  stockAskSize,
					long**	  stockLastSize,

					double**	 stockBidDate,
					double**	 stocknAskDate,
					double**	 stockLastDate,

					long**	  stockVolume,

					long**		stockFstOptIdx,
					long**		stockLstOptIdx,
					long*		  stocksCount);

extern "C" 
bool InitApplication();


extern "C" 
void AppendDataToOptionMessage(
															 long						optionID,
															 long						stockID,
															 float					preIV,
															 float					IV,
															 float					bidPrice,
															 float					askPrice,
															 float					lastPrice,
															 long						bidSize,
															 long						askSize,
															 long						lastSize,
															 double					bidDate,
															 double					askDate,
															 double					lastDate,
															 long						volume,
															 const GREEKS&	greeks);

extern "C" 
bool SendOptionMessage();


extern "C" 
void AppendDataToStockMessage(
															 long						stockID,
															 float					bidPrice,
															 float					askPrice,
															 float					lastPrice,
															 long						bidSize,
															 long						askSize,
															 long						lastSize,
															 double					bidDate,
															 double					askDate,
															 double					lastDate,
															 long						volume);

extern "C" 
bool SendStockMessage();



//bool ParseOptionFile(const std::string& strFileName, OptionList& optionList);


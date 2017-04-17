#include "stdafx.h"
#include "Logger.h"
#include <stdio.h>
#include <stdarg.h>

void WriteToLog(const char* szFormat, ...)
{


	char buf[2048];

	va_list pArgPtr;
	va_start(pArgPtr, szFormat);

	vsprintf(
		buf,
		szFormat,
		pArgPtr); 

	static FILE* f = fopen("OptionCalcGPU.log", "w");
	fprintf(f, "%s", buf);	

}

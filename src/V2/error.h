/*********************************************************************
 * error.h
 *********************************************************************/

#ifndef _ERROR_H_
#define _ERROR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdarg.h>

	void KLTError(char* fmt, ...);
	void KLTWarning(char* fmt, ...);

#ifdef __cplusplus
}
#endif
#endif



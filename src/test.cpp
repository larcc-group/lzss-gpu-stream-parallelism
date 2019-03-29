#include <string.h>
#include "md5.h"
 
int main() {
    unsigned char digest[16];
    const char* string = "happy";
    printDigestMD5(string,strlen(string));
    return 0;
}
#include <string.h>
#include <openssl/md5.h>
#include <iostream>
#include <stdio.h>
 
void printDigestMD5(const char* buffer, size_t size) {
    unsigned char digest[16];
 
    printf("string length: %d\n", size); 
 
    MD5_CTX ctx;
    MD5_Init(&ctx);
    MD5_Update(&ctx, buffer, size);
    MD5_Final(digest, &ctx);
 
    char mdString[33];
    for (int i = 0; i < 16; i++)
        sprintf(&mdString[i*2], "%02x", (unsigned int)digest[i]);
 
    printf("md5 digest: %s\n", mdString);
 
}
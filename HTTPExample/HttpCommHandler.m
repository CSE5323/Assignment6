//
//  HttpCommHandler.m
//
//
//

#import "HttpCommHandler.h"

@implementation HttpCommHandler

NSString * const SUCCESSFUL_UPLOAD_RESPONSE = @"Success";
NSString * const FAILED_UPLOAD_RESPONSE = @"Failure";


//-(NSString*) sendPostRequestTo: (NSURL*) url
//                      withBody: (NSData*) requestBody
//         withCompletionHandler: ()
//{
//    NSLog(@"HTTP Request");
//    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
//    NSLog(@"Request URL: %@",[url absoluteString]);
//    NSLog(@"Request Body: %@",[[NSString alloc] initWithData:requestBody encoding:NSUTF8StringEncoding]);
//    [request setHTTPMethod:@"POST"];
//    [request setHTTPBody:requestBody];
//    NSURLResponse *response = NULL;
//    NSError *requestError = NULL;
//    NSData *responseData = [NSURLConnection sendSynchronousRequest:request returningResponse:&response error:&requestError];
//    NSString *responseString = [[NSString alloc] initWithData:responseData encoding:NSUTF8StringEncoding];
//    NSLog(@"Response: %@",responseString);
//    return responseString;
//}

-(NSString*) sendPostRequestTo: (NSURL*) url withBody: (NSData*) requestBody
{
    NSLog(@"HTTP Request");
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    NSLog(@"Request URL: %@",[url absoluteString]);
    NSLog(@"Request Body: %@",[[NSString alloc] initWithData:requestBody encoding:NSUTF8StringEncoding]);
    [request setHTTPMethod:@"POST"];
    [request setHTTPBody:requestBody];
    NSURLResponse *response = NULL;
    NSError *requestError = NULL;
    NSData *responseData = [NSURLConnection sendSynchronousRequest:request returningResponse:&response error:&requestError];
    NSString *responseString = [[NSString alloc] initWithData:responseData encoding:NSUTF8StringEncoding];
    NSLog(@"Response: %@",responseString);
    return responseString;
}

-(NSString*) sendTempPostRequestTo: (NSURL*) url withBody: (NSData*) requestBody
{
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    [request setHTTPMethod:@"POST"];
    [request setHTTPBody:requestBody];
    NSURLResponse *response = NULL;
    NSError *requestError = NULL;
    NSData *responseData = [NSURLConnection sendSynchronousRequest:request returningResponse:&response error:&requestError];
    NSString *responseString = [[NSString alloc] initWithData:responseData encoding:NSUTF8StringEncoding];
    return responseString;
    
}

-(NSDictionary *) getTupleFromJSONString: (NSString *)responseString
{
    NSData *responseData = [responseString dataUsingEncoding:NSUTF8StringEncoding];
    NSError *e = nil;
    NSDictionary *JSON = [NSJSONSerialization JSONObjectWithData: responseData options: NSJSONReadingMutableContainers error: &e];
    return JSON;
}


-(NSString*) sendPostWithData:(NSData *)data at: (NSURL *)url withFileName: (NSString*) fileName andServerSessionID: (NSString*) serverSessionID
{
    
    // note that nginx needs to be load balancing for this to work properly
    NSMutableURLRequest *request = [[NSMutableURLRequest alloc] init];
    NSLog(@"Request URL: %@",[url absoluteString]);
    [request setURL:url];
    [request setHTTPMethod:@"POST"];
    int dataLength = (int)[data length];
    NSMutableData *requestMutableBody = [[NSMutableData alloc]init];
    [requestMutableBody setData:data];

    NSString* contentDisp = [NSString stringWithFormat:@"attachment; filename=\"%@.txt\"",fileName];
    [request addValue:[NSString stringWithFormat:@"%d",dataLength] forHTTPHeaderField:@"Content-Length"];
    [request addValue:@"application/octet-stream" forHTTPHeaderField:@"Content-Type"];
    [request addValue:contentDisp forHTTPHeaderField:@"Content-Disposition"];
    [request addValue:[NSString stringWithFormat:@"bytes 0-%d/%d",(dataLength-1),dataLength] forHTTPHeaderField:@"X-Content-Range"];
    [request setHTTPBody:requestMutableBody];
    
    NSLog(@"Request Headers: %@", [request allHTTPHeaderFields]);
    NSLog(@"filename is the name with which you want the file to be saved\ncontent length is the size of the data in bytes\ncontent type will remain octet stream only\nsession id is pid_sid_tid and x-content-range is the sequence of bytes you are uploading. we can upload the data in chunks. so x-y/z means the current chunk is bytes xth to yth and the total number of bytes in the total data is z. so for a 100 byte data, with one chunk (always use one chunk right now), the header will be 0-99/100. in case you have two chunks of 50 bytes each, then the first will be 0-49/100 and second: 50-99/100");
    NSLog(@"Request body is the byte stream of zipped temporal data");
    NSData *returnData = [NSURLConnection sendSynchronousRequest:request returningResponse:nil error:nil];
    NSString *returnString = [[NSString alloc] initWithData:returnData encoding:NSUTF8StringEncoding];
    NSLog(@"--%@",returnString);
    
    return returnString;

}

-(NSString*) sendPostWithSplitData:(NSData *)data at: (NSURL *)url withFileName: (NSString*) fileName andServerSessionID: (NSString*) serverSessionID andStartByte:(int) sByte andEndByte:(int)eByte andSize:(int)dataLength
{
    NSMutableURLRequest *request = [[NSMutableURLRequest alloc] init];
    [request setURL:url];
    [request setHTTPMethod:@"POST"];
    
//    NSString *contentLength = [NSString stringWithFormat:@"Content-Length: %d\r\n", dataLength];
    
    
//    NSData *requestBody=[contentLength dataUsingEncoding:NSUTF8StringEncoding];
    NSMutableData *requestMutableBody = [[NSMutableData alloc]init];
    [requestMutableBody setData:data];
    NSString* contentDisp = [NSString stringWithFormat:@"attachment; filename=\"%@\"",fileName];
    
    [request addValue:[NSString stringWithFormat:@"%d",dataLength] forHTTPHeaderField:@"Content-Length"];
    [request addValue:@"application/octet-stream" forHTTPHeaderField:@"Content-Type"];
    [request addValue:contentDisp forHTTPHeaderField:@"Content-Disposition"];
    [request addValue:[NSString stringWithFormat:@"bytes %d-%d/%d",sByte,eByte,dataLength] forHTTPHeaderField:@"X-Content-Range"];
    [request addValue:serverSessionID forHTTPHeaderField:@"Session-ID"]; //pid_sid_tid
    NSLog(@"%@", [request allHTTPHeaderFields]);
    //    NSLog(@"%@",[request valueForHTTPHeaderField:field]);
    
    
    [request setHTTPBody:requestMutableBody];
    
    NSHTTPURLResponse *response;
    NSData *returnData = [NSURLConnection sendSynchronousRequest:request returningResponse:&response error:nil];
    NSString *returnString = [[NSString alloc] initWithData:returnData encoding:NSUTF8StringEncoding];
    
    NSLog(@"Response Code>> %ld",(long)[response statusCode]);
    
    NSLog(@"----> %@",returnString);
    
    
    if ([response statusCode] == 200 || [response statusCode] == 201)
    {
        returnString = SUCCESSFUL_UPLOAD_RESPONSE;
    }
    else
    {
        returnString = FAILED_UPLOAD_RESPONSE;
    }
    
    return returnString;
}


+(BOOL)uploadSuccessfulBasedOn:(NSString *)responseString
{
    return (responseString == SUCCESSFUL_UPLOAD_RESPONSE);
}


@end

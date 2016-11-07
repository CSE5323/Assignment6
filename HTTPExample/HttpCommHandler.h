//
//  HttpCommHandler.h


#import <Foundation/Foundation.h>

@interface HttpCommHandler : NSObject
-(NSString*) sendPostRequestTo: (NSURL*) url withBody: (NSData*) requestBody;
-(NSString*) sendTempPostRequestTo: (NSURL*) url withBody: (NSData*) requestBody;
-(NSString*) sendPostWithData:(NSData *)data at: (NSURL *)url withFileName: (NSString*) fileName andServerSessionID: (NSString*) serverSessionID;
-(NSDictionary *) getTupleFromJSONString: (NSString *)responseString;
-(NSString*) sendPostWithSplitData:(NSData *)data at: (NSURL *)url withFileName: (NSString*) fileName andServerSessionID: (NSString*) serverSessionID andStartByte:(int) sByte andEndByte:(int)eByte andSize:(int)dataLength;
+(BOOL)uploadSuccessfulBasedOn:(NSString *)responseString;
@end

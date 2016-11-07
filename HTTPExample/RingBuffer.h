//
//  RingBuffer.h
//  HTTPExample
//
//  Copyright (c) 2014 Eric Larson. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface RingBuffer : NSObject

-(NSArray*)getDataAsVector;
-(void) addNewData:(float)xData withY:(float)yData withZ:(float)zData;
@end

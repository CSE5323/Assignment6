//
//  RingBuffer.m
//  HTTPExample
//
//  Copyright (c) 2014 Eric Larson. All rights reserved.
//

#import "RingBuffer.h"
#define BUFFER_SIZE 10

@interface RingBuffer()
{
    float x[BUFFER_SIZE];
    float y[BUFFER_SIZE];
    float z[BUFFER_SIZE];
}

@property int head;

@end

@implementation RingBuffer

-(id)init{
    self = [super init];
    if(self){
        self.head = 0;
    }
    return self;
}

-(void) addNewData:(float)xData
             withY:(float)yData
             withZ:(float)zData
{
    x[self.head] = xData;
    y[self.head] = yData;
    z[self.head] = zData;
    
    [self incrementHead];
    
}

-(NSArray*)getDataAsVector{
    NSMutableArray* vectorData = [[NSMutableArray alloc]initWithCapacity:3*BUFFER_SIZE];
    
    for(int i=0;i<BUFFER_SIZE;++i){
        int index = (self.head+i)%BUFFER_SIZE;
        vectorData[3*i] = @(x[index]);
        vectorData[3*i+1] = @(y[index]);
        vectorData[3*i+2] = @(z[index]);
    }
    
    return vectorData;
}

-(void)incrementHead{
    self.head++;
    if(self.head >= BUFFER_SIZE)
        self.head = 0;
}

@end

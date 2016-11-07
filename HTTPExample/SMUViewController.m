//
//  SMUViewController.m
//  HTTPExample
//
//  Copyright (c) 2014 Eric Larson. All rights reserved.
//

#import "SMUViewController.h"
#import <CoreMotion/CoreMotion.h>
#import "RingBuffer.h"

// CHANGE THIS URL TO THAT OF YOUR SERVER!!!!
// if you do not know your local sharing server name try:
//    ifconfig |grep inet
// to see what your public facing IP address is, the ip address can be used here
#define SERVER_URL "http://10.8.123.119:8000"   // or the name might be in format: "http://erics-macbook-pro.local:8000"
#define UPDATE_INTERVAL 1/10.0

@interface SMUViewController () <NSURLSessionTaskDelegate>

// for the machine learning session
@property (strong,nonatomic) NSURLSession *session;
@property (strong,nonatomic) NSNumber *dsid;
@property (weak, nonatomic) IBOutlet UILabel *dsidLabel;

// display views
@property (weak, nonatomic) IBOutlet UIView *topRight;
@property (weak, nonatomic) IBOutlet UIView *bottomRight;
@property (weak, nonatomic) IBOutlet UIView *topLeft;
@property (weak, nonatomic) IBOutlet UIView *bottomLeft;

// for storing accelerometer updates
@property (strong, nonatomic) CMMotionManager *cmMotionManager;
@property (strong, nonatomic) NSOperationQueue *backQueue;
@property (strong, nonatomic) RingBuffer *ringBuffer;

@property (atomic) float magValue;
@property (atomic) BOOL isCalibrating;
@property (atomic) BOOL isWaitingForInputData;
@property (atomic) NSInteger calibrationStage;
// 0        1
// 2        3

@end

@implementation SMUViewController

#pragma mark - Getter Overloads
-(CMMotionManager*)cmMotionManager{
    if(!_cmMotionManager){
        _cmMotionManager = [[CMMotionManager alloc] init];
        
        if(![_cmMotionManager isDeviceMotionAvailable])
            _cmMotionManager = nil;
        else
            _cmMotionManager.deviceMotionUpdateInterval = UPDATE_INTERVAL;
    }
    return _cmMotionManager;
}

-(NSOperationQueue*)backQueue{
    
    if(!_backQueue){
        _backQueue = [[NSOperationQueue alloc] init];
    }
    return _backQueue;
}

-(RingBuffer*)ringBuffer{
    if(!_ringBuffer){
        _ringBuffer = [[RingBuffer alloc] init];
    }
    
    return _ringBuffer;
}


#pragma mark - View Controller Life Cycle Methods
- (void)viewDidLoad
{
    [super viewDidLoad];
	// Do any additional setup after loading the view, typically from a nib.
    
    _dsid = @1;
    _isCalibrating = NO;
    _isWaitingForInputData = YES;
    _calibrationStage = -1;
    _magValue = 0.1;
    
    //setup NSURLSession (ephemeral)
    NSURLSessionConfiguration *sessionConfig =
    [NSURLSessionConfiguration ephemeralSessionConfiguration];
    
    sessionConfig.timeoutIntervalForRequest = 5.0;
    sessionConfig.timeoutIntervalForResource = 8.0;
    sessionConfig.HTTPMaximumConnectionsPerHost = 1;
    
    self.session =
    [NSURLSession sessionWithConfiguration:sessionConfig
                                  delegate:self
                             delegateQueue:nil];
    
    // setup acceleration monitoring
    [self.cmMotionManager startDeviceMotionUpdatesToQueue:self.backQueue withHandler:^(CMDeviceMotion *motion, NSError *error) {
        [_ringBuffer addNewData:motion.userAcceleration.x
                          withY:motion.userAcceleration.y
                          withZ:motion.userAcceleration.z];
        float mag = fabs(motion.userAcceleration.x)+fabs(motion.userAcceleration.y)+fabs(motion.userAcceleration.z);

        if(mag > self.magValue){ // do this and return immediately
            [self.backQueue addOperationWithBlock:^{
                [self motionEventOccurred];
            }];
        }
    }];

}

-(void)dealloc{
    [self.cmMotionManager stopDeviceMotionUpdates];
}

#pragma mark - IBActions
- (IBAction)sliderChanged:(UISlider*)sender {
    self.magValue = sender.value;
}

- (IBAction)startCalibration:(id)sender {
    self.isCalibrating = YES;
    [self nextCalibrationStage];
}

#pragma mark - Calibration and segmentation methods
-(void)motionEventOccurred{
    if(self.isCalibrating){
        //send a labeled example
        if(self.calibrationStage >= 0 && self.isWaitingForInputData)
        {
            self.isWaitingForInputData = NO;
            // send data to the server with label
            [self sendFeatureArray:[self.ringBuffer getDataAsVector]
                         withLabel:@(self.calibrationStage)];
            [self nextCalibrationStage];
        }
    }
    else
    {
        if(self.isWaitingForInputData)
        {
            self.isWaitingForInputData = NO;
            //predict a label
            [self predictFeature:[self.ringBuffer getDataAsVector]];
            [self performSelector:@selector(setWaitingToTrue) withObject:nil afterDelay:1.0];
        }
    }
}

-(void)setWaitingToTrue{
    self.isWaitingForInputData = YES;
}

-(void)nextCalibrationStage{
    switch (self.calibrationStage) {
        case -1:
        {
            //start in top left
            dispatch_async(dispatch_get_main_queue(), ^{
                [self makeRed:self.topLeft];
                self.calibrationStage=0;
                [self performSelector:@selector(setWaitingToTrue) withObject:nil afterDelay:1.0];
            });
            break;
        }
        case 0:
        {
            //move to top right
            dispatch_async(dispatch_get_main_queue(), ^{
                [self makeBlue:self.topLeft];
                [self makeRed:self.topRight];
                self.calibrationStage=1;
                [self performSelector:@selector(setWaitingToTrue) withObject:nil afterDelay:1.0];
            });
            break;
        }
        case 1:
        {
            //move to bottom right
            dispatch_async(dispatch_get_main_queue(), ^{
                [self makeBlue:self.topRight];
                [self makeRed:self.bottomRight];
                self.calibrationStage=2;
                [self performSelector:@selector(setWaitingToTrue) withObject:nil afterDelay:1.0];
            });
            break;
        }
        case 2:
        {
            //move to bottom left
            dispatch_async(dispatch_get_main_queue(), ^{
                [self makeBlue:self.bottomRight];
                [self makeRed:self.bottomLeft];
                self.calibrationStage=3;
                [self performSelector:@selector(setWaitingToTrue) withObject:nil afterDelay:1.0];
            });
            break;
        }
        case 3:
        {
            //stop calibration
            dispatch_async(dispatch_get_main_queue(), ^{
                [self makeBlue:self.bottomLeft];
                self.isCalibrating = NO;
                self.calibrationStage=-1;
                [self performSelector:@selector(setWaitingToTrue) withObject:nil afterDelay:1.0];
            });
            break;
        }
        default:
            break;
    }
    
}

#pragma mark - touch response feedback methods
- (void) startFlashToView:(UIView*)view
{
    [self performSelector:@selector(makeRed:) withObject:view afterDelay: 0];
    [self performSelector:@selector(makeBlue:) withObject:view afterDelay: 0.5];
    [self performSelector:@selector(makeRed:) withObject:view afterDelay: 1.0];
    [self performSelector:@selector(makeBlue:) withObject:view afterDelay: 1.5];
    
}

-(void)makeBlue:(UIView*)view{
    [view setBackgroundColor:[UIColor blueColor]];
}

-(void)makeRed:(UIView*)view{
    [view setBackgroundColor:[UIColor redColor]];
}


#pragma mark - HTTP Post and Get Request Methods
- (IBAction)getDataSetId:(id)sender {
    
    // get a new dataset ID from the server (gives back a new dataset id)
    // Note that if data is not uploaded, the server may issue the same dsid to another requester
    // ---how might you solve this problem?---
    
    // create a GET request and get the reponse back as NSData
    NSString *baseURL = [NSString stringWithFormat:@"%s/GetNewDatasetId",SERVER_URL];

    NSURL *getUrl = [NSURL URLWithString: baseURL];
    NSURLSessionDataTask *dataTask = [self.session dataTaskWithURL:getUrl
     completionHandler:^(NSData *data,
                         NSURLResponse *response,
                         NSError *error) {
         if(!error){
             NSLog(@"%@",response);
             NSDictionary *responseData = [NSJSONSerialization JSONObjectWithData:data options: NSJSONReadingMutableContainers error: &error];
             self.dsid = responseData[@"dsid"];
             NSLog(@"New dataset id is %@",self.dsid);
             
             dispatch_async(dispatch_get_main_queue(), ^{
                 self.dsidLabel.text = [NSString stringWithFormat:@"DSID: %ld",(long)[self.dsid integerValue]];
             });
         } else{
             NSLog(@"%@",error);
         }
     }];
    [dataTask resume]; // start the task
    
}

- (IBAction)updateModel:(id)sender {
    // tell the server to train a new model for the given dataset id (dsid)
    
    // create a GET request and get the reponse back as NSData
    NSString *baseURL = [NSString stringWithFormat:@"%s/UpdateModel",SERVER_URL];
    NSString *query = [NSString stringWithFormat:@"?dsid=%d",[self.dsid intValue]];
    
    NSURL *getUrl = [NSURL URLWithString: [baseURL stringByAppendingString:query]];
    NSURLSessionDataTask *dataTask = [self.session dataTaskWithURL:getUrl
         completionHandler:^(NSData *data,
                             NSURLResponse *response,
                             NSError *error) {
             if(!error){
                 // we should get back the accuracy of the model
                 NSLog(@"%@",response);
                 NSDictionary *responseData = [NSJSONSerialization JSONObjectWithData:data options: NSJSONReadingMutableContainers error: &error];
                 NSLog(@"Accuracy using resubstitution: %@",responseData[@"resubAccuracy"]);
             }
         }];
    [dataTask resume]; // start the task
}

- (void)predictFeature:(NSArray*)featureData {
    // send the server new feature data and request back a prediction of the class
    
    // setup the url
    NSString *baseURL = [NSString stringWithFormat:@"%s/PredictOne",SERVER_URL];
    NSURL *postUrl = [NSURL URLWithString:baseURL];
    
    
    // data to send in body of post request (send arguments as json)
    NSError *error = nil;
    NSDictionary *jsonUpload = @{@"feature":featureData,
                                 @"dsid":self.dsid};
    
    NSData *requestBody=[NSJSONSerialization dataWithJSONObject:jsonUpload options:NSJSONWritingPrettyPrinted error:&error];
    
    // create a custom HTTP POST request
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:postUrl];
    
    [request setHTTPMethod:@"POST"];
    [request setHTTPBody:requestBody];
    
    // start the request, print the responses etc.
    NSURLSessionDataTask *postTask = [self.session dataTaskWithRequest:request
         completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
             if(!error){
                 NSDictionary *responseData = [NSJSONSerialization JSONObjectWithData:data options: NSJSONReadingMutableContainers error: &error];
                 
                 NSString *labelResponse = [NSString stringWithFormat:@"%@",[responseData valueForKey:@"prediction"]];
                 NSLog(@"%@",labelResponse);
                 
                 dispatch_async(dispatch_get_main_queue(), ^{
                     // flash the view that was touched
                     if([labelResponse  isEqual: @"[0]"])
                         [self startFlashToView:self.topLeft];
                     else if([labelResponse  isEqual: @"[1]"])
                             [self startFlashToView:self.topRight];
                     else if([labelResponse  isEqual: @"[2]"])
                         [self startFlashToView:self.bottomRight];
                     else if([labelResponse  isEqual: @"[3]"])
                         [self startFlashToView:self.bottomLeft];
                     
                     self.isWaitingForInputData = YES;
                 });
             }
         }];
    [postTask resume];
}

- (void)sendFeatureArray:(NSArray*)data
               withLabel:(NSNumber*)label
{
    // Add a data point and a label to the database for the current dataset ID
    
    // setup the url
    NSString *baseURL = [NSString stringWithFormat:@"%s/AddDataPoint",SERVER_URL];
    NSURL *postUrl = [NSURL URLWithString:baseURL];
    
    
    // make an array of feature data
    // and place inside a dictionary with the label and dsid
    NSError *error = nil;
    NSDictionary *jsonUpload = @{@"feature":data,
                                 @"label":label,
                                 @"dsid":self.dsid};
    
    NSData *requestBody=[NSJSONSerialization dataWithJSONObject:jsonUpload options:NSJSONWritingPrettyPrinted error:&error];
    
    // create a custom HTTP POST request
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:postUrl];
    
    [request setHTTPMethod:@"POST"];
    [request setHTTPBody:requestBody];
    
    // start the request, print the responses etc.
    NSURLSessionDataTask *postTask = [self.session dataTaskWithRequest:request
     completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
         if(!error){
             NSLog(@"%@",response);
             NSDictionary *responseData = [NSJSONSerialization JSONObjectWithData:data options: NSJSONReadingMutableContainers error: &error];
             
             // we should get back the feature data from the server and the label it parsed
             NSString *featuresResponse = [NSString stringWithFormat:@"%@",[responseData valueForKey:@"feature"]];
             NSString *labelResponse = [NSString stringWithFormat:@"%@",[responseData valueForKey:@"label"]];
             NSLog(@"received %@ and %@",featuresResponse,labelResponse);
         }
     }];
    [postTask resume];
    
}

@end

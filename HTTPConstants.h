//
//  HTTPConstants.h
//  HTTPExample
//
//  Created by Eric Larson on 1/7/14.
//  Copyright (c) 2014 Eric Larson. All rights reserved.
//

#ifndef HTTPExample_HTTPConstants_h
#define HTTPExample_HTTPConstants_h

//HTTP Status Codes
#define HTTP_OK 200
#define HTTP_CREATED 201
#define HTTP_ACCEPTED 202
#define HTTP_NO_CONTENT 204
#define HTTP_BAD_REQUEST 400
#define HTTP_UNAUTHORIZED 401
#define HTTP_PAYMENT_REQUIRED 402
#define HTTP_FORBIDDEN 403
#define HTTP_NOT_FOUND 404
#define HTTP_METHOD_NOT_ALLOWED 405
#define HTTP_LENGTH_REQUIRED 411
#define HTTP_INTERNAL_SERVER_ERROR 500

//RESOURCE Status Codes.
//A resource can be a data source, dataset, model or prediction
#define WAITING 0
#define QUEUED 1
#define STARTED 2
#define IN_PROGRESS 3
#define SUMMARIZED 4
#define FINISHED 5
#define FAULTY -1
#define UNKNOWN -2
#define RUNNABLE -3

#endif

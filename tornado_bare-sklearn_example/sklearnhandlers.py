#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pickle
from bson.binary import Binary
import json
import numpy as np

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = [float(val) for val in vals]
        label = data['label']
        sess  = data['dsid']

        dbid = self.db.labeledinstances.insert(
            {"feature":fvals,"label":label,"dsid":sess}
            );
        self.write_json({"id":str(dbid),"feature":fvals,"label":label})

class SetParams(BaseHandler):
    def post(self):
        data = json.loads(self.request.body.decode("utf-8"))
        self.application.knn_param = data['KNeighbors']
        self.application.svm_C_param = data['SVM']

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class UpdateModelForDatasetId(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        # dsid = self.get_int_arg("dsid", default = 0)

        # create feature vectors from database
        f=[];
        for a in self.db.labeledinstances.find({}):
            f.append([float(val) for val in a['feature']])

        # create label vector from database
        l=[];
        for a in self.db.labeledinstances.find({}):
            l.append(a['label'])

        # fit the model to the data
        c1 = KNeighborsClassifier(n_neighbors = self.knn_param)
        c2 = svm.svc(gamma = 0.0, C = self.svm_C_param)

        acc_knn = -1;
        acc_svm = -1;

        if l:
            c1.fit(f,l) # training
            c2.fit(f,l)

            lstar_knn = c1.predict(f)
            lstar_svm = c2.predict(f)

            self.clf["KNeighbors"] = c1
            self.clf["SVM"] = c2

            acc_knn = sum(lstar_knn == l)/float(len(l))
            acc_svm = sum(lstar_svm == l)/float(len(l))

            bytes_knn = pickle.dumps(c1)
            bytes_svm = pickle.dumps(c2)

            self.db.models.update({"classifier": "KNeighbors"},
                {  "$set": {"model":Binary(bytes_knn)}  },
                upsert=True)

            self.db.models.update({"classifier": "SVM"},
                {  "$set": {"model":Binary(bytes_svm)}  },
                upsert=True)

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy_knn":acc_knn, "resubAccuracy_svm":acc_svm})

class PredictOneFromDatasetId(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature'];
        fvals = [float(val) for val in vals];
        fvals = np.array(fvals).reshape(1, -1)
        # dsid  = data['dsid']

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!

        if self.clf:
            print('Loading Model From DB')
            pred_label_knn = self.clf["KNeighbors"].predict(fvals);
            pred_label_svm = self.clf["SVM"].predict(fvals);
            self.write_json({"prediction_knn": str(pred_label_knn), "prediction_svm": str(pred_label_svm)})
        else:
            raise HTTPError(status_code = 404, log_message = "No model!")

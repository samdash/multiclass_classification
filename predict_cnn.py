import tensorflow as tf
from data_helpers import sentence_to_index
import pickle
import json

from operator import itemgetter
from flask import Flask, jsonify
with open ('./checkpoints/vocab_shape.pickle', 'rb') as fp:
    vocabulary,shape = pickle.load(fp)

labels = json.loads(open('./checkpoints/labels.json').read())
print(labels)

checkpoint_file = tf.train.latest_checkpoint('./checkpoints')


print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_text").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        scores_norm = graph.get_operation_by_name("output/scores_norm").outputs[0]


def predict():
    #text = 'how to restart my apple watch?'
    #text = 'My account has been charged two times please fix the problem and refund my account'
    text = 'I can’t seem to find how to sign in with my YouTube account into the apple tv YouTube app'
    raw_x = sentence_to_index(text, vocabulary, shape)
    predicted_scores = sess.run(scores_norm, {input_x: raw_x, dropout_keep_prob: 1.0})
    #print(predicted_scores[0])
    res = []
    output = {}
    for i,prediction in enumerate(predicted_scores[0]):
        #print(i,str(format(prediction,'.10f')))
        res.append({"tag": labels[i], "score": str(format(prediction,'0.10f'))})
    #print(labels[20],str(format(predicted_scores[0][20])))
    newlist = sorted(res, key=itemgetter('score'), reverse=True)
    print(newlist[0:3])
    output['tag_details'] = {"response_time": "10", "tags": newlist[0:3]} # return only top 3 tags
    print(output)


if __name__ == '__main__':
    predict()

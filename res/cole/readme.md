# there is no stopping the cole train

this folder is basically for me to mess around with tf.keras until it successfully trains on the dataset we are using

### Notes from Issac

- data passed to learn is always a list of dictionaries
  - each dict contains the correct result
- data passed to evaluate is a list of dictionaries
  - each dict does not contain the 'correct' key

## Notes on working parts

- sklearnsvm.py is working completely, with k-fold and holdout
- tensorsvm.py works with holdout, never got k-fold working
- TensorflowTest.py works with holdout, tests NN

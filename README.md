# Getting started

  ## Installation

  This package is written in Python 3.9.12. Packages are forked from [this GitHub repository](https://github.com/PiotrZiolo/recommender-systems-class) from university classes. This project was focused on making content based recommender for hotel, to recommend best items to user.

  ### Install dependencies

  All dependencies with corresponding versions are listed in `requirements.txt`. To install them all at once you can use the following command:

  ```bash
  pip install -r requirements.txt
  ```

  ### Run project in jupyter notebook

  Project is run in a Jupyter notebook. To run it you can use the following command:

  ```bash
  jupyter notebook
  ```

  You can also run the project in Visual Studio Code using jupyter extension. This extension is available in the [VSCode marketplace](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

  ### Generating `.html` files from jupyter notebook

  To generate `.html` files from jupyter notebook you can use the following commands:
  ```bash
   jupyter nbconvert --to html project_1_data_preparation.ipynb
   jupyter nbconvert --to html project_2_recommender_and_evaluation.ipynb
  ```

  Project is separated into two notebooks. First one is used to prepare data for recommender. Second one is used to train and evaluate recommender. So it is needed to run both notebooks to get the results.

# Results

  The best result was achieved after some adjustments in tuning function and is on an approximate level of `HR@10: 0.252885`.

  <img src="assets/final_result.png" />

  The best parameters of the recommender:
  ```js
  {
    n_neg_per_pos = 9,
    epoch_count = 460,
    lr = 0.000086,
    wd = 0.0001,
    hidden_1 = 330,
    hidden_2 = 30
  }
  ```

  The result was better than attached in this repository `AmazonRecommender`. Comparsion:
  <img src="assets/final_comparsion.png" />

# Further improvement

  The result can be better with some more user and item features. I was also experimenting with some different approach to generating negative interactions. In my method each room have assigned some alternate solutions and from this list the negative interactions are generated. In this way we can prevent adding negative interaction to to the room, which could be very likely selected by the user. This approach unfortunately was too computationally complex to perform in a reasonable amount of time and my implementation was also showing unsatisfactory results. I am also thinking about the neural network model which could be changed to some MLP or GMF implementations.
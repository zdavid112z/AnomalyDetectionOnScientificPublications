# Anomaly Detection on Scientific Publication

Algorithm for performing author identification on scientific publications. For a given author and its past papers, determine the confidence that a new publication belongs to them.

The general approach is that we pass their publications' abstracts through a transformer to get their embeddings. We then average those embeddings, and this vector represents a user profile. For our queried publication, we pass its abstract through a transformer and the resulted embeddings vector is compared with the user profile. If the similarity level is above a certain threshold, then we determine the publications is similar to the author's past publications. The similarity level can also be used to obtain a percentual confidence level.

The purpose of this tool is to quickly reject publications clearly outside one's area of expertise, thus the main objective is minimizing the number of false negatives. Overall, the solution obtains 91% accuracy.

[Presentation](Ion%20David-Gabriel%20-%20Anomaly%20Detection%20on%20Scientific%20Publications.pdf)

[Documentation](Diploma%20Project.pdf)

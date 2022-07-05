# match-job-posts-with-university-courses

NLP thesis project

Abstract:
During university enrollment, students have the difficult task of choosing electives or specialization courses that give them enough knowledge and expertise to fulfill the requirements of
desired job positions. This thesis addresses this challenge by using Natural Language Processing (NLP) models that does the matching between jobs and courses and can potentially help
students to select course more accurately based on their career hopes. For this study, we use
Information Technology (IT) job postings data from the Stack Overflow job ad platform and
extract a list of courses from the IT University of Copenhagen (ITU). We pose this task as a
multi-label classification task where a model predicts a set of possible courses to take for a
given IT job. To obtain labels, we manually annotate unique job titles for courses to take as test
and use distant supervision for automatic annotation for training and development. We extract
the most relevant information from job postings and use state-of-the-art language models such
as BERT (Devlin et al., 2018), RoBERTa (Liu et al., 2019) and JobBERT (Zhang et al., 2022) to
predict and evaluate the most relevant courses for a given job.

Models: BERT, RoBERTa, JOBBERT, Word2Vec, TF-IDF, Logistic Regression

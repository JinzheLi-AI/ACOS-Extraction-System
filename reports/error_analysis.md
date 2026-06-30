# ACOS Error Analysis

**Prediction file:** `C:\Users\Administrator\OneDrive\Desktop\Semester 2 2025-2026\FYP1\Code\course-acos-extraction\data\predictions\predictions.csv`

**Total evaluated examples:** 986
**Total gold tuples:** 1444
**Total predicted tuples:** 1238
**Total exact matched tuples:** 310

## Error Type Counts

| Error Type | Count | Explanation |
| --- | ---: | --- |
| `exact_match` | 310 | The predicted tuple exactly matches a gold tuple across all four fields. |
| `missed_tuple` | 260 | A gold tuple was not matched by any predicted tuple. |
| `hallucinated_tuple` | 54 | The model predicted a tuple that does not correspond to any gold tuple. |
| `wrong_sentiment` | 150 | The aspect/category/opinion may partly match, but the sentiment label is wrong. |
| `wrong_category` | 301 | The model used the wrong ACOS category for a partially matched tuple. |
| `wrong_opinion` | 571 | The model generated an incorrect or different opinion phrase. |
| `wrong_aspect` | 341 | The model generated an incorrect aspect or missed an implicit aspect. |
| `partial_match` | 874 | At least one field matched, but the full quadruple was not exactly correct. |
| `malformed_output` | 0 | The model produced non-empty text that could not be parsed as an ACOS tuple. |

## Representative Examples

Examples are limited to a few cases per error type to keep the report readable.

### exact_match

**Example 1**

- **Input text:** The weekly assignments were challenging and a lot of fun .
- **Gold output:** `(assignments \| assignments quality \| challenging and a lot of fun \| positive)`
- **Model prediction:** `(assignments \| assignments quality \| challenging and a lot of fun \| positive)`
- **Gold tuple:** `(assignments \| assignments quality \| challenging and a lot of fun \| positive)`
- **Predicted tuple:** `(assignments \| assignments quality \| challenging and a lot of fun \| positive)`
- **Matched fields:** aspect, category, opinion, sentiment
- **Wrong fields:** N/A

**Example 2**

- **Input text:** I think I learned a great deal from this course and I ' m looking forward to the specialisation courses .
- **Gold output:** `(course \| course general \| learned a great deal \| positive)`
- **Model prediction:** `(course \| course general \| learned a great deal \| positive)`
- **Gold tuple:** `(course \| course general \| learned a great deal \| positive)`
- **Predicted tuple:** `(course \| course general \| learned a great deal \| positive)`
- **Matched fields:** aspect, category, opinion, sentiment
- **Wrong fields:** N/A

**Example 3**

- **Input text:** This is simply a great course .
- **Gold output:** `(course \| course general \| simply a great \| positive)`
- **Model prediction:** `(course \| course general \| simply a great \| positive)`
- **Gold tuple:** `(course \| course general \| simply a great \| positive)`
- **Predicted tuple:** `(course \| course general \| simply a great \| positive)`
- **Matched fields:** aspect, category, opinion, sentiment
- **Wrong fields:** N/A


### missed_tuple

**Example 1**

- **Input text:** Well pased and full of information .
- **Gold output:** `(implicit \| course quality \| Well pased \| positive) ; (implicit \| course quality \| full \| positive)`
- **Model prediction:** `(implicit \| course comprehensiveness \| Well passed and full of information \| positive)`
- **Gold tuple:** `(implicit \| course quality \| full \| positive)`
- **Predicted tuple:** `N/A`
- **Matched fields:** N/A
- **Wrong fields:** N/A

**Example 2**

- **Input text:** The instructor is easy to understand and makes a good job introducing the language .
- **Gold output:** `(instructor \| faculty comprehensiveness \| easy to understand \| positive) ; (instructor \| faculty general \| makes a good job \| positive)`
- **Model prediction:** `(instructor \| faculty comprehensiveness \| easy to understand \| positive)`
- **Gold tuple:** `(instructor \| faculty general \| makes a good job \| positive)`
- **Predicted tuple:** `N/A`
- **Matched fields:** N/A
- **Wrong fields:** N/A

**Example 3**

- **Input text:** The lectures were great and engaging .
- **Gold output:** `(lectures \| presentation quality \| great \| positive) ; (lectures \| presentation quality \| engaging \| positive)`
- **Model prediction:** `(lectures \| presentation quality \| great and engaging \| positive)`
- **Gold tuple:** `(lectures \| presentation quality \| engaging \| positive)`
- **Predicted tuple:** `N/A`
- **Matched fields:** N/A
- **Wrong fields:** N/A


### hallucinated_tuple

**Example 1**

- **Input text:** The last week of the course also introduces an interesting and seemingly effective framework for design thinking .
- **Gold output:** `(course \| course general \| introduces an interesting and seemingly effective \| positive)`
- **Model prediction:** `(course \| course general \| interesting \| positive) ; (course \| course quality \| implicit \| positive)`
- **Gold tuple:** `N/A`
- **Predicted tuple:** `(course \| course quality \| implicit \| positive)`
- **Matched fields:** N/A
- **Wrong fields:** N/A

**Example 2**

- **Input text:** I would recommend it to be more data oriented like those case studies with specific examples so that it naturally converges to creating case study for a city at the end of the course as final project rather than optional study .
- **Gold output:** `(implicit \| course quality \| implicit \| negative)`
- **Model prediction:** `(course \| course general \| recommend \| positive) ; (course \| course general \| implicit \| positive)`
- **Gold tuple:** `N/A`
- **Predicted tuple:** `(course \| course general \| recommend \| positive)`
- **Matched fields:** N/A
- **Wrong fields:** N/A

**Example 3**

- **Input text:** This course was difficult to follow and the free trials offered seemed forced upon us .
- **Gold output:** `(course \| course comprehensiveness \| difficult to follow \| negative)`
- **Model prediction:** `(course \| course comprehensiveness \| difficult to follow \| negative) ; (implicit \| course general \| implicit \| negative)`
- **Gold tuple:** `N/A`
- **Predicted tuple:** `(implicit \| course general \| implicit \| negative)`
- **Matched fields:** N/A
- **Wrong fields:** N/A


### wrong_sentiment

**Example 1**

- **Input text:** Prepare yourself though - it is quite a lot of material .
- **Gold output:** `(material \| material quantity \| quite a lot \| negative)`
- **Model prediction:** `(material \| material quality \| very a lot \| positive)`
- **Gold tuple:** `(material \| material quantity \| quite a lot \| negative)`
- **Predicted tuple:** `(material \| material quality \| very a lot \| positive)`
- **Matched fields:** aspect
- **Wrong fields:** category, opinion, sentiment

**Example 2**

- **Input text:** I felt like they went too fast .
- **Gold output:** `(implicit \| course general \| too fast \| positive)`
- **Model prediction:** `(implicit \| food general \| implicit \| negative)`
- **Gold tuple:** `(implicit \| course general \| too fast \| positive)`
- **Predicted tuple:** `(implicit \| food general \| implicit \| negative)`
- **Matched fields:** aspect
- **Wrong fields:** category, opinion, sentiment

**Example 3**

- **Input text:** The reason that I ' ve only rated 3 stars however is that the only useful thing you learn in this course is the design thinking framework .
- **Gold output:** `(course \| course quality \| only rated 3 stars \| negative)`
- **Model prediction:** `(course \| course quality \| implicit \| positive)`
- **Gold tuple:** `(course \| course quality \| only rated 3 stars \| negative)`
- **Predicted tuple:** `(course \| course quality \| implicit \| positive)`
- **Matched fields:** aspect, category
- **Wrong fields:** opinion, sentiment


### wrong_category

**Example 1**

- **Input text:** Well pased and full of information .
- **Gold output:** `(implicit \| course quality \| Well pased \| positive) ; (implicit \| course quality \| full \| positive)`
- **Model prediction:** `(implicit \| course comprehensiveness \| Well passed and full of information \| positive)`
- **Gold tuple:** `(implicit \| course quality \| well pased \| positive)`
- **Predicted tuple:** `(implicit \| course comprehensiveness \| well passed and full of information \| positive)`
- **Matched fields:** aspect, sentiment
- **Wrong fields:** category, opinion

**Example 2**

- **Input text:** Prepare yourself though - it is quite a lot of material .
- **Gold output:** `(material \| material quantity \| quite a lot \| negative)`
- **Model prediction:** `(material \| material quality \| very a lot \| positive)`
- **Gold tuple:** `(material \| material quantity \| quite a lot \| negative)`
- **Predicted tuple:** `(material \| material quality \| very a lot \| positive)`
- **Matched fields:** aspect
- **Wrong fields:** category, opinion, sentiment

**Example 3**

- **Input text:** I felt like they went too fast .
- **Gold output:** `(implicit \| course general \| too fast \| positive)`
- **Model prediction:** `(implicit \| food general \| implicit \| negative)`
- **Gold tuple:** `(implicit \| course general \| too fast \| positive)`
- **Predicted tuple:** `(implicit \| food general \| implicit \| negative)`
- **Matched fields:** aspect
- **Wrong fields:** category, opinion, sentiment


### wrong_opinion

**Example 1**

- **Input text:** Changed the way I think about several things which is what I think education should be about .
- **Gold output:** `(implicit \| course general \| Changed the way I think \| positive)`
- **Model prediction:** `(implicit \| course general \| implicit \| positive)`
- **Gold tuple:** `(implicit \| course general \| changed the way i think \| positive)`
- **Predicted tuple:** `(implicit \| course general \| implicit \| positive)`
- **Matched fields:** aspect, category, sentiment
- **Wrong fields:** opinion

**Example 2**

- **Input text:** I greatly enjoyed this course .
- **Gold output:** `(course \| course general \| greatly enjoyed \| positive)`
- **Model prediction:** `(course \| course general \| very enjoyed \| positive)`
- **Gold tuple:** `(course \| course general \| greatly enjoyed \| positive)`
- **Predicted tuple:** `(course \| course general \| very enjoyed \| positive)`
- **Matched fields:** aspect, category, sentiment
- **Wrong fields:** opinion

**Example 3**

- **Input text:** Well pased and full of information .
- **Gold output:** `(implicit \| course quality \| Well pased \| positive) ; (implicit \| course quality \| full \| positive)`
- **Model prediction:** `(implicit \| course comprehensiveness \| Well passed and full of information \| positive)`
- **Gold tuple:** `(implicit \| course quality \| well pased \| positive)`
- **Predicted tuple:** `(implicit \| course comprehensiveness \| well passed and full of information \| positive)`
- **Matched fields:** aspect, sentiment
- **Wrong fields:** category, opinion


### wrong_aspect

**Example 1**

- **Input text:** As an overview , this class works well .
- **Gold output:** `(class \| course general \| works well \| positive)`
- **Model prediction:** `(implicit \| course general \| well \| positive)`
- **Gold tuple:** `(class \| course general \| works well \| positive)`
- **Predicted tuple:** `(implicit \| course general \| well \| positive)`
- **Matched fields:** category, sentiment
- **Wrong fields:** aspect, opinion

**Example 2**

- **Input text:** Excellent tutors and material .
- **Gold output:** `(tutors \| faculty general \| Excellent \| positive) ; (material \| material quality \| Excellent \| positive)`
- **Model prediction:** `(course \| course general \| Excellent \| positive) ; (material \| material quality \| Excellent \| positive)`
- **Gold tuple:** `(tutors \| faculty general \| excellent \| positive)`
- **Predicted tuple:** `(course \| course general \| excellent \| positive)`
- **Matched fields:** opinion, sentiment
- **Wrong fields:** aspect, category

**Example 3**

- **Input text:** I do see value in including this material in the specialisation but not as an outright course .
- **Gold output:** `(course \| course general \| implicit \| negative)`
- **Model prediction:** `(material \| material quality \| implicit \| negative)`
- **Gold tuple:** `(course \| course general \| implicit \| negative)`
- **Predicted tuple:** `(material \| material quality \| implicit \| negative)`
- **Matched fields:** opinion, sentiment
- **Wrong fields:** aspect, category


### partial_match

**Example 1**

- **Input text:** Changed the way I think about several things which is what I think education should be about .
- **Gold output:** `(implicit \| course general \| Changed the way I think \| positive)`
- **Model prediction:** `(implicit \| course general \| implicit \| positive)`
- **Gold tuple:** `(implicit \| course general \| changed the way i think \| positive)`
- **Predicted tuple:** `(implicit \| course general \| implicit \| positive)`
- **Matched fields:** aspect, category, sentiment
- **Wrong fields:** opinion

**Example 2**

- **Input text:** I greatly enjoyed this course .
- **Gold output:** `(course \| course general \| greatly enjoyed \| positive)`
- **Model prediction:** `(course \| course general \| very enjoyed \| positive)`
- **Gold tuple:** `(course \| course general \| greatly enjoyed \| positive)`
- **Predicted tuple:** `(course \| course general \| very enjoyed \| positive)`
- **Matched fields:** aspect, category, sentiment
- **Wrong fields:** opinion

**Example 3**

- **Input text:** Well pased and full of information .
- **Gold output:** `(implicit \| course quality \| Well pased \| positive) ; (implicit \| course quality \| full \| positive)`
- **Model prediction:** `(implicit \| course comprehensiveness \| Well passed and full of information \| positive)`
- **Gold tuple:** `(implicit \| course quality \| well pased \| positive)`
- **Predicted tuple:** `(implicit \| course comprehensiveness \| well passed and full of information \| positive)`
- **Matched fields:** aspect, sentiment
- **Wrong fields:** category, opinion

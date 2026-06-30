# Final Model Error Analysis

**Prediction file:** `C:\Users\Administrator\Documents\Codex\2026-05-31\files-mentioned-by-the-user-acos\course-acos-extraction\data\predictions\predictions_final.csv`

**Total evaluated examples:** 986
**Total gold tuples:** 1444
**Total predicted tuples:** 1278
**Total exact matched tuples:** 427

## Error Type Counts

| Error Type | Count | Explanation |
| --- | ---: | --- |
| `exact_match` | 427 | The predicted tuple exactly matches a gold tuple across all four fields. |
| `missed_tuple` | 220 | A gold tuple was not matched by any predicted tuple. |
| `hallucinated_tuple` | 54 | The model predicted a tuple that does not correspond to any gold tuple. |
| `wrong_sentiment` | 145 | The aspect/category/opinion may partly match, but the sentiment label is wrong. |
| `wrong_category` | 299 | The model used the wrong ACOS category for a partially matched tuple. |
| `wrong_opinion` | 508 | The model generated an incorrect or different opinion phrase. |
| `wrong_aspect` | 235 | The model generated an incorrect aspect or missed an implicit aspect. |
| `partial_match` | 797 | At least one field matched, but the full quadruple was not exactly correct. |
| `malformed_output` | 0 | The model produced non-empty text that could not be parsed as an ACOS tuple. |

## Representative Examples

Examples are limited to a few cases per error type to keep the report readable.

### exact_match

**Example 1**

- **Input text:** I greatly enjoyed this course .
- **Gold output:** `(course \| course general \| greatly enjoyed \| positive)`
- **Model prediction:** `(course \| course general \| greatly enjoyed \| positive)`
- **Gold tuple:** `(course \| course general \| greatly enjoyed \| positive)`
- **Predicted tuple:** `(course \| course general \| greatly enjoyed \| positive)`
- **Matched fields:** aspect, category, opinion, sentiment
- **Wrong fields:** N/A

**Example 2**

- **Input text:** The instructor is easy to understand and makes a good job introducing the language .
- **Gold output:** `(instructor \| faculty comprehensiveness \| easy to understand \| positive) ; (instructor \| faculty general \| makes a good job \| positive)`
- **Model prediction:** `(instructor \| faculty comprehensiveness \| easy to understand \| positive) ; (instructor \| faculty general \| good job introducing the language \| positive)`
- **Gold tuple:** `(instructor \| faculty comprehensiveness \| easy to understand \| positive)`
- **Predicted tuple:** `(instructor \| faculty comprehensiveness \| easy to understand \| positive)`
- **Matched fields:** aspect, category, opinion, sentiment
- **Wrong fields:** N/A

**Example 3**

- **Input text:** Prepare yourself though - it is quite a lot of material .
- **Gold output:** `(material \| material quantity \| quite a lot \| negative)`
- **Model prediction:** `(material \| material quantity \| quite a lot \| negative)`
- **Gold tuple:** `(material \| material quantity \| quite a lot \| negative)`
- **Predicted tuple:** `(material \| material quantity \| quite a lot \| negative)`
- **Matched fields:** aspect, category, opinion, sentiment
- **Wrong fields:** N/A


### missed_tuple

**Example 1**

- **Input text:** I learned a whole lot and grasped and retained more only because of his teaching style and methods .
- **Gold output:** `(implicit \| course general \| learned a whole lot \| positive) ; (implicit \| faculty general \| grasped and retained more \| positive)`
- **Model prediction:** `(implicit \| faculty comprehensiveness \| implicit \| positive) ; (implicit \| faculty comprehensiveness \| implicit \| positive)`
- **Gold tuple:** `(implicit \| faculty general \| grasped and retained more \| positive)`
- **Predicted tuple:** `N/A`
- **Matched fields:** N/A
- **Wrong fields:** N/A

**Example 2**

- **Input text:** I do see value in including this material in the specialisation but not as an outright course .
- **Gold output:** `(course \| course general \| implicit \| negative)`
- **Model prediction:** `(material \| material quality \| value \| positive)`
- **Gold tuple:** `(course \| course general \| implicit \| negative)`
- **Predicted tuple:** `N/A`
- **Matched fields:** N/A
- **Wrong fields:** N/A

**Example 3**

- **Input text:** The amount of work and time that went in to creating this comprehensive and diverse course is extremely impressive .
- **Gold output:** `(course \| course workload \| extremely impressive \| positive) ; (course \| course quality \| diverse \| positive) ; (course \| course comprehensiveness \| comprehensive \| positive)`
- **Model prediction:** `(course \| course quality \| comprehensive and diverse \| positive) ; (course \| course workload \| extremely impressive \| positive)`
- **Gold tuple:** `(course \| course comprehensiveness \| comprehensive \| positive)`
- **Predicted tuple:** `N/A`
- **Matched fields:** N/A
- **Wrong fields:** N/A


### hallucinated_tuple

**Example 1**

- **Input text:** The weekly assignments were challenging and a lot of fun .
- **Gold output:** `(assignments \| assignments quality \| challenging and a lot of fun \| positive)`
- **Model prediction:** `(assignments \| assignments quality \| challenging \| positive) ; (assignments \| assignments quality \| lot of fun \| positive)`
- **Gold tuple:** `N/A`
- **Predicted tuple:** `(assignments \| assignments quality \| lot of fun \| positive)`
- **Matched fields:** N/A
- **Wrong fields:** N/A

**Example 2**

- **Input text:** The last week of the course also introduces an interesting and seemingly effective framework for design thinking .
- **Gold output:** `(course \| course general \| introduces an interesting and seemingly effective \| positive)`
- **Model prediction:** `(course \| course general \| interesting \| positive) ; (course \| course quality \| implicit \| positive)`
- **Gold tuple:** `N/A`
- **Predicted tuple:** `(course \| course quality \| implicit \| positive)`
- **Matched fields:** N/A
- **Wrong fields:** N/A

**Example 3**

- **Input text:** I do see value in including this material in the specialisation but not as an outright course .
- **Gold output:** `(course \| course general \| implicit \| negative)`
- **Model prediction:** `(material \| material quality \| value \| positive)`
- **Gold tuple:** `N/A`
- **Predicted tuple:** `(material \| material quality \| value \| positive)`
- **Matched fields:** N/A
- **Wrong fields:** N/A


### wrong_sentiment

**Example 1**

- **Input text:** I felt like they went too fast .
- **Gold output:** `(implicit \| course general \| too fast \| positive)`
- **Model prediction:** `(implicit \| food general \| went too fast \| negative)`
- **Gold tuple:** `(implicit \| course general \| too fast \| positive)`
- **Predicted tuple:** `(implicit \| food general \| went too fast \| negative)`
- **Matched fields:** aspect
- **Wrong fields:** category, opinion, sentiment

**Example 2**

- **Input text:** The reason that I ' ve only rated 3 stars however is that the only useful thing you learn in this course is the design thinking framework .
- **Gold output:** `(course \| course quality \| only rated 3 stars \| negative)`
- **Model prediction:** `(course \| course quality \| implicit \| positive)`
- **Gold tuple:** `(course \| course quality \| only rated 3 stars \| negative)`
- **Predicted tuple:** `(course \| course quality \| implicit \| positive)`
- **Matched fields:** aspect, category
- **Wrong fields:** opinion, sentiment

**Example 3**

- **Input text:** Challenging Assignments .
- **Gold output:** `(Assignments \| assignments quality \| Challenging \| positive)`
- **Model prediction:** `(Assignments \| assignments quality \| Challenging \| negative)`
- **Gold tuple:** `(assignments \| assignments quality \| challenging \| positive)`
- **Predicted tuple:** `(assignments \| assignments quality \| challenging \| negative)`
- **Matched fields:** aspect, category, opinion
- **Wrong fields:** sentiment


### wrong_category

**Example 1**

- **Input text:** I felt like they went too fast .
- **Gold output:** `(implicit \| course general \| too fast \| positive)`
- **Model prediction:** `(implicit \| food general \| went too fast \| negative)`
- **Gold tuple:** `(implicit \| course general \| too fast \| positive)`
- **Predicted tuple:** `(implicit \| food general \| went too fast \| negative)`
- **Matched fields:** aspect
- **Wrong fields:** category, opinion, sentiment

**Example 2**

- **Input text:** He has really taken a great approach to presenting this course .
- **Gold output:** `(implicit \| faculty comprehensiveness \| really taken a great approach \| positive)`
- **Model prediction:** `(implicit \| faculty general \| great \| positive)`
- **Gold tuple:** `(implicit \| faculty comprehensiveness \| really taken a great approach \| positive)`
- **Predicted tuple:** `(implicit \| faculty general \| great \| positive)`
- **Matched fields:** aspect, sentiment
- **Wrong fields:** category, opinion

**Example 3**

- **Input text:** I learned a whole lot and grasped and retained more only because of his teaching style and methods .
- **Gold output:** `(implicit \| course general \| learned a whole lot \| positive) ; (implicit \| faculty general \| grasped and retained more \| positive)`
- **Model prediction:** `(implicit \| faculty comprehensiveness \| implicit \| positive) ; (implicit \| faculty comprehensiveness \| implicit \| positive)`
- **Gold tuple:** `(implicit \| course general \| learned a whole lot \| positive)`
- **Predicted tuple:** `(implicit \| faculty comprehensiveness \| implicit \| positive)`
- **Matched fields:** aspect, sentiment
- **Wrong fields:** category, opinion


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

- **Input text:** The weekly assignments were challenging and a lot of fun .
- **Gold output:** `(assignments \| assignments quality \| challenging and a lot of fun \| positive)`
- **Model prediction:** `(assignments \| assignments quality \| challenging \| positive) ; (assignments \| assignments quality \| lot of fun \| positive)`
- **Gold tuple:** `(assignments \| assignments quality \| challenging and a lot of fun \| positive)`
- **Predicted tuple:** `(assignments \| assignments quality \| challenging \| positive)`
- **Matched fields:** aspect, category, sentiment
- **Wrong fields:** opinion

**Example 3**

- **Input text:** I think I learned a great deal from this course and I ' m looking forward to the specialisation courses .
- **Gold output:** `(course \| course general \| learned a great deal \| positive)`
- **Model prediction:** `(course \| course general \| great \| positive)`
- **Gold tuple:** `(course \| course general \| learned a great deal \| positive)`
- **Predicted tuple:** `(course \| course general \| great \| positive)`
- **Matched fields:** aspect, category, sentiment
- **Wrong fields:** opinion


### wrong_aspect

**Example 1**

- **Input text:** The course design is thoughtful and well - carried - out .
- **Gold output:** `(course design \| course quality \| thoughtful \| positive) ; (course design \| course quality \| well - carried - out \| positive)`
- **Model prediction:** `(course \| course quality \| thoughtful \| positive) ; (course \| course quality \| well - carried - out \| positive)`
- **Gold tuple:** `(course design \| course quality \| thoughtful \| positive)`
- **Predicted tuple:** `(course \| course quality \| thoughtful \| positive)`
- **Matched fields:** category, opinion, sentiment
- **Wrong fields:** aspect

**Example 2**

- **Input text:** The course design is thoughtful and well - carried - out .
- **Gold output:** `(course design \| course quality \| thoughtful \| positive) ; (course design \| course quality \| well - carried - out \| positive)`
- **Model prediction:** `(course \| course quality \| thoughtful \| positive) ; (course \| course quality \| well - carried - out \| positive)`
- **Gold tuple:** `(course design \| course quality \| well - carried - out \| positive)`
- **Predicted tuple:** `(course \| course quality \| well - carried - out \| positive)`
- **Matched fields:** category, opinion, sentiment
- **Wrong fields:** aspect

**Example 3**

- **Input text:** The illustrations are delightful !
- **Gold output:** `(illustrations \| presentation quality \| delightful \| positive)`
- **Model prediction:** `(implicit \| presentation quality \| illustrations are delightful \| positive)`
- **Gold tuple:** `(illustrations \| presentation quality \| delightful \| positive)`
- **Predicted tuple:** `(implicit \| presentation quality \| illustrations are delightful \| positive)`
- **Matched fields:** category, sentiment
- **Wrong fields:** aspect, opinion


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

- **Input text:** The weekly assignments were challenging and a lot of fun .
- **Gold output:** `(assignments \| assignments quality \| challenging and a lot of fun \| positive)`
- **Model prediction:** `(assignments \| assignments quality \| challenging \| positive) ; (assignments \| assignments quality \| lot of fun \| positive)`
- **Gold tuple:** `(assignments \| assignments quality \| challenging and a lot of fun \| positive)`
- **Predicted tuple:** `(assignments \| assignments quality \| challenging \| positive)`
- **Matched fields:** aspect, category, sentiment
- **Wrong fields:** opinion

**Example 3**

- **Input text:** I think I learned a great deal from this course and I ' m looking forward to the specialisation courses .
- **Gold output:** `(course \| course general \| learned a great deal \| positive)`
- **Model prediction:** `(course \| course general \| great \| positive)`
- **Gold tuple:** `(course \| course general \| learned a great deal \| positive)`
- **Predicted tuple:** `(course \| course general \| great \| positive)`
- **Matched fields:** aspect, category, sentiment
- **Wrong fields:** opinion

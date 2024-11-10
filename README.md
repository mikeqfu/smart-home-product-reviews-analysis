Understanding user experience using text mining & analysis
==========================================================

<span style="font-size: large; ">**A case study of Amazon.com reviews for smart-home products - Technical Documentation**</span>

Authors: Qian Fu [![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAARCAYAAAA7bUf6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAV5JREFUOI2tlDFIw0AUhv9XiyAUqSKCIFgHF0HodYou7ezWMXZQcbXg7qqjIDiL6aAd3dzE2KGNS6+DFMTBFjqIiLTSSZDn4CUm14ZG8Z8u793/8e7duxA0nUtji5g2AeS0lM3EpYJwLN1D7qIsjRQzXQJI65s0NYg4bwqn5QZiPoAEkDZFFStzO3h974ZB0swky9JIBSCqguSICvxKKs/3cVQPztyAv4KZyR/u+FgCH5/9AImJtwvCseKqiZ6K2Sbun09x83AEU1QDpk6vgrv2gQdTXiuGwVsI6OR2GWW5huvHXUxNLCEzv+dP5wDVkyh66dfhtA+xOL0+kIsMcUHD9CvIbCITCrGjAoyFfTy9XfnDNgDEmbhETLkwczHb9NadXgX1zrH3zcQlQI39RX1VYvS462psZGoCUD0h4jyA0Dkfoq7ywIOYwmkRsQDQiFIBEQv/AyR9x19+Bf+iL7V0glIITNCxAAAAAElFTkSuQmCC)](https://orcid.org/0000-0002-6502-9934), Yixiu Yu [![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAARCAYAAAA7bUf6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAV5JREFUOI2tlDFIw0AUhv9XiyAUqSKCIFgHF0HodYou7ezWMXZQcbXg7qqjIDiL6aAd3dzE2KGNS6+DFMTBFjqIiLTSSZDn4CUm14ZG8Z8u793/8e7duxA0nUtji5g2AeS0lM3EpYJwLN1D7qIsjRQzXQJI65s0NYg4bwqn5QZiPoAEkDZFFStzO3h974ZB0swky9JIBSCqguSICvxKKs/3cVQPztyAv4KZyR/u+FgCH5/9AImJtwvCseKqiZ6K2Sbun09x83AEU1QDpk6vgrv2gQdTXiuGwVsI6OR2GWW5huvHXUxNLCEzv+dP5wDVkyh66dfhtA+xOL0+kIsMcUHD9CvIbCITCrGjAoyFfTy9XfnDNgDEmbhETLkwczHb9NadXgX1zrH3zcQlQI39RX1VYvS462psZGoCUD0h4jyA0Dkfoq7ywIOYwmkRsQDQiFIBEQv/AyR9x19+Bf+iL7V0glIITNCxAAAAAElFTkSuQmCC)](https://orcid.org/0000-0002-3481-0648), Dong Zhang [![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAARCAYAAAA7bUf6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAV5JREFUOI2tlDFIw0AUhv9XiyAUqSKCIFgHF0HodYou7ezWMXZQcbXg7qqjIDiL6aAd3dzE2KGNS6+DFMTBFjqIiLTSSZDn4CUm14ZG8Z8u793/8e7duxA0nUtji5g2AeS0lM3EpYJwLN1D7qIsjRQzXQJI65s0NYg4bwqn5QZiPoAEkDZFFStzO3h974ZB0swky9JIBSCqguSICvxKKs/3cVQPztyAv4KZyR/u+FgCH5/9AImJtwvCseKqiZ6K2Sbun09x83AEU1QDpk6vgrv2gQdTXiuGwVsI6OR2GWW5huvHXUxNLCEzv+dP5wDVkyh66dfhtA+xOL0+kIsMcUHD9CvIbCITCrGjAoyFfTy9XfnDNgDEmbhETLkwczHb9NadXgX1zrH3zcQlQI39RX1VYvS462psZGoCUD0h4jyA0Dkfoq7ywIOYwmkRsQDQiFIBEQv/AyR9x19+Bf+iL7V0glIITNCxAAAAAElFTkSuQmCC)](https://orcid.org/0000-0002-0993-207X)


## Introduction

This study examines the growing use of smart home products, focusing on user experiences beyond initial adoption. By analysing Amazon.com reviews of robotic vacuum cleaners, it aimed to uncover what drives user satisfaction and dissatisfaction.

🔍 Key insights include:

- ✔️ Satisfaction dimensions: Users value functionality, smart capabilities and enhanced performance.
- ❌ Dissatisfaction dimensions: Common issues include limited "smartness", poor customer service and functionality issues (e.g. connectivity). 

💡 Notably, the concept of "smartness" emerges as a double-edged sword - contributing to satisfaction when effective, yet leading to disappointment when poorly implemented.

Through fuzzy-set qualitative comparative analysis (fsQCA), this study offers a comprehensive framework for understanding key user experience dimensions. The methods and insights may be valuable for designers and marketers across various smart home product categories, such as electric vehicles, to inform product design and strategy.


## Methodology

<!--suppress HtmlDeprecatedAttribute -->
<p align="center">
    <img src="docs/source/_images/methodology/text_mining_framework.svg" width="600" height="796" alt="Text-mining framework applied in this study."/>
</p>
<p align="center">
  <em>Fig. 1: Text-mining framework applied in this study. (<a href="https://doi.org/10.1080/08874417.2024.2408006">Yu, et al., 2024</a>)</em>
</p>

**For more details, please refer to the full [Technical Documentation](https://github.com/mikeqfu/smart-home-product-reviews-analysis/blob/master/docs/build/latex/smart_home_product_reviews_analysis.pdf)**.


## Publications

- [Yu, Y.](https://orcid.org/0000-0002-3481-0648), [Fu, Q.](https://orcid.org/0000-0002-6502-9934), [Zhang, D.](https://orcid.org/0000-0002-0993-207X), & [Gu, Q.](https://orcid.org/0000-0001-6049-4282) (2024). Understanding user experience with smart home products. Journal of Computer Information Systems, 1–23. [doi:10.1080/08874417.2024.2408006](https://doi.org/10.1080/08874417.2024.2408006).
- [Yu, Y.](https://orcid.org/0000-0002-3481-0648), [Fu, Q.](https://orcid.org/0000-0002-6502-9934), [Zhang, D.](https://orcid.org/0000-0002-0993-207X) & [Gu, Q.](https://orcid.org/0000-0001-6049-4282) (2024). What are smart home product users commenting on? A case study of robotic vacuums. In: Han, H., Baker, E. (eds) Next Generation Data Science. SDSC 2023. Communications in Computer and Information Science, vol 2113. Springer, Cham. [doi:10.1007/978-3-031-61816-1_3](https://doi.org/10.1007/978-3-031-61816-1_3).


## Relevant resources

- **Open-source Python packages:** [**Fu, Q.**](https://research.birmingham.ac.uk/en/persons/qian-fu) (2020). [PyHelpers](https://pypi.org/project/pyhelpers/): an open-source toolkit for facilitating Python users' data manipulation tasks. [doi:10.5281/zenodo.4081634](https://doi.org/10.5281/zenodo.4017438).


## Collaborators

<table>
    <tbody>
        <tr>
            <td align="center">
                <a href="https://github.com/mikeqfu" target="_blank"><img src="https://avatars.githubusercontent.com/u/1729711?v=4?s=100" width="100px;" alt="Qian Fu"/><br><sub><b>Qian Fu</b></sub></a><br>
                <a href="https://github.com/mikeqfu/smart-home-product-reviews-analysis/commits?author=mikeqfu" target="_blank" title="Methodology, Software">💻</a>
                <a href="https://github.com/mikeqfu/smart-home-product-reviews-analysis/tree/master/tests" target="_blank" title="Validation">🧪</a>
                <a href="https://github.com/mikeqfu/smart-home-product-reviews-analysis/tree/master/demos" target="_blank" title="Data Curation, Visualisation">📈</a>
                <a href="https://github.com/mikeqfu/smart-home-product-reviews-analysis/blob/master/docs/build/latex/smart_home_product_reviews_analysis.pdf" target="_blank" title="Documentation">📚</a>
                <a href="https://doi.org/10.1080/08874417.2024.2408006" target="_blank" title="Writing - Review & Editing">📝</a>
            </td>
            &ensp;
            <td align="center">
                <a href="https://github.com/ashleyashley2022" target="_blank"><img src="https://avatars.githubusercontent.com/u/96884205?v=4?s=100" width="100px;" alt="Yixiu Yu"/><br><sub><b>Yixiu Yu</b></sub></a><br>
                <a href="https://orcid.org/0000-0002-3481-0648" target="_blank" title="Conceptualization & Methodology">💡</a>
                <a title="Resources, Data Curation">📦</a>
                <a title="Investigation">🔍</a>
                <a href="https://doi.org/10.1080/08874417.2024.2408006" target="_blank" title="Validation, Formal analysis">📊</a>
                <a href="https://doi.org/10.1080/08874417.2024.2408006" target="_blank" title="Writing - Original Draft, Writing - Review & Editing">📝</a>
            </td>
            &ensp;
            <td align="center">
                <a href="https://github.com/danbaidong" target="_blank"><img src="https://avatars.githubusercontent.com/u/4456514?v=4?s=100" width="100px;" alt="Dong Zhang"/><br><sub><b>Dong Zhang</b></sub></a><br>
                <a href="https://orcid.org/0000-0002-0993-207X" target="_blank" title="Initiation">🌱</a>
                <a title="Conceptualization & Methodology">💡</a>
                <a title="Formal analysis">📊</a>
                <a href="https://doi.org/10.1080/08874417.2024.2408006" target="_blank" title="Writing - Review & Editing">📝</a>
            </td>
        </tr>
    </tbody>
</table>
<br>

<span style="font-size: large; ">**_Disclaimer_**</span>

<em><span style="font-size: small; ">The data used in this study consists of anonymous reviews sourced from Amazon.com and is intended strictly for research purposes. All data has been processed and analyzed without personal identifiers to ensure the anonymity of the reviewers. Due to privacy and confidentiality agreements, we are not permitted to share any specific data or individual reviews included in our analysis. While the findings of this study are based on information gathered from these reviews, we acknowledge that individual opinions reflect the personal experiences of customers. Therefore, we cannot guarantee the reliability or validity of the content within the reviews. All data usage complies with Amazon's terms of service and privacy policies. The authors of this research and the collaborators of this repository have no affiliation with Amazon.com and do not receive any financial support or benefits from the platform.</span></em>

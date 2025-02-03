import pandas as pd
import json
from datasets import Dataset
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
import re

def preprocess_arabic(text: str) -> str:
    """
    Basic preprocessing for Arabic text
    """
    text = str(text)
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Remove tatweel
    text = re.sub(r'\u0640', '', text)
    # Normalize alef
    text = re.sub(r'[إأٱآا]', 'ا', text)
    # Normalize hamza
    text = re.sub(r'[ؤئ]', 'ء', text)
    # Normalize teh marbuta
    text = re.sub(r'ة', 'ه', text)
    # Normalize yeh
    text = re.sub(r'[يى]', 'ي', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def generate_qa_pairs(verse: str, translation: str, context: str, surah: int, ayah: int) -> List[Dict]:
    """
    Generate question-answer pairs for a given verse with focus on real-world applications and scholarly interpretations
    """
    qa_pairs = []
    
    # Basic verse identification
    qa_pairs.append({
        "question": f"ماذا تقول الآية {ayah} من سورة {surah}؟",  # What does verse {ayah} of surah {surah} say?
        "context": context,
        "answer_text": verse,
        "answer_start": context.find(verse),
        "verse": verse,
        "translation": translation,
        "surah": surah,
        "ayah": ayah,
        "question_type": "verse_content"
    })
    
    # Ethical guidance questions
    ethical_questions = [
        f"كيف يمكن تطبيق تعاليم الآية {ayah} من سورة {surah} في حياتنا اليومية؟",  # How can we apply the teachings of this verse in our daily life?
        f"ما هو الموقف الأخلاقي الذي تقدمه الآية {ayah} من سورة {surah}؟",  # What ethical stance does this verse present?
        f"كيف يمكن أن تساعدنا هذه الآية في اتخاذ القرارات الأخلاقية؟"  # How can this verse help us make ethical decisions?
    ]
    
    for q in ethical_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "ethical_guidance"
        })
    
    # Contemporary issues questions
    contemporary_questions = [
        f"كيف يمكن فهم هذه الآية في سياق التحديات المعاصرة؟",  # How can we understand this verse in the context of contemporary challenges?
        f"ما هو الإرشاد الذي تقدمه هذه الآية للمجتمع الحديث؟",  # What guidance does this verse offer to modern society?
        f"كيف يمكن تطبيق هذه الآية في عصر التكنولوجيا والعولمة؟"  # How can this verse be applied in the age of technology and globalization?
    ]
    
    for q in contemporary_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "contemporary_application"
        })
    
    # Social relations questions
    social_questions = [
        f"كيف تساعدنا هذه الآية في تحسين علاقاتنا مع الآخرين؟",  # How does this verse help us improve our relationships with others?
        f"ما هو منظور هذه الآية حول التعامل مع المجتمع؟",  # What is this verse's perspective on dealing with society?
        f"كيف يمكن تطبيق هذه الآية في حل النزاعات الاجتماعية؟"  # How can this verse be applied in resolving social conflicts?
    ]
    
    for q in social_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "social_relations"
        })
    
    # Personal development questions
    personal_questions = [
        f"كيف يمكن أن تساعدنا هذه الآية في تطوير أنفسنا؟",  # How can this verse help us in self-development?
        f"ما هي الدروس الشخصية التي يمكن استخلاصها من هذه الآية؟",  # What personal lessons can be drawn from this verse?
        f"كيف يمكن تطبيق هذه الآية في تحسين سلوكنا اليومي؟"  # How can this verse be applied to improve our daily behavior?
    ]
    
    for q in personal_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "personal_development"
        })
    
    # Interpretation and scholarly discussion questions
    interpretation_questions = [
        # Different schools of thought
        f"كيف فسر العلماء المختلفون هذه الآية {ayah} من سورة {surah}؟",  # How have different scholars interpreted this verse?
        f"ما هي وجهات النظر المختلفة في تفسير هذه الآية؟",  # What are the different viewpoints in interpreting this verse?
        
        # Historical context vs modern interpretation
        f"كيف يختلف فهم هذه الآية في سياقها التاريخي عن تفسيرها المعاصر؟",  # How does the historical understanding of this verse differ from contemporary interpretation?
        
        # Reconciliation questions
        f"كيف يمكن التوفيق بين هذه الآية وبين المفاهيم العلمية الحديثة؟",  # How can this verse be reconciled with modern scientific concepts?
        f"كيف نفهم هذه الآية في ضوء التنوع الثقافي والديني في العالم المعاصر؟",  # How do we understand this verse in light of cultural and religious diversity in the contemporary world?
        
        # Complex application questions
        f"كيف نطبق هذه الآية في المواقف التي تتعارض فيها القيم التقليدية مع المتطلبات المعاصرة؟",  # How do we apply this verse in situations where traditional values conflict with contemporary requirements?
        f"كيف يمكن فهم هذه الآية في سياق القضايا الأخلاقية المعقدة في عصرنا؟",  # How can we understand this verse in the context of complex ethical issues in our time?
        
        # Specific controversial topics
        f"كيف تساعدنا هذه الآية في فهم العلاقة بين العلم والدين؟",  # How does this verse help us understand the relationship between science and religion?
        f"كيف نفهم هذه الآية في سياق حقوق الإنسان والمساواة؟",  # How do we understand this verse in the context of human rights and equality?
        f"ما هو دور هذه الآية في الحوار بين الأديان والثقافات المختلفة؟"  # What is the role of this verse in interfaith and intercultural dialogue?
    ]
    
    for q in interpretation_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "interpretation_discussion"
        })
    
    # Madhab and scholarly interpretation questions
    madhab_questions = [
        # Specific madhab interpretations
        f"كيف فسر المذهب الحنفي هذه الآية، وكيف تختلف عن تفسير المذهب الشافعي؟",  # How did the Hanafi madhab interpret this verse, and how does it differ from the Shafi'i interpretation?
        f"ما هو موقف المذهب المالكي من تفسير هذه الآية، وكيف يقارن بالمذهب الحنبلي؟",  # What is the Maliki madhab's stance on interpreting this verse, and how does it compare to the Hanbali view?
        f"كيف تناول علماء المذاهب الأربعة الرئيسية تفسير هذه الآية؟",  # How did scholars of the four main madhabs approach the interpretation of this verse?
        
        # Historical debates
        f"ما هي أبرز المناقشات التاريخية بين العلماء حول تفسير هذه الآية؟",  # What are the most prominent historical discussions among scholars about interpreting this verse?
        f"كيف تطور فهم هذه الآية عبر العصور الإسلامية المختلفة؟",  # How has the understanding of this verse evolved through different Islamic eras?
        
        # Usul al-Fiqh perspectives
        f"كيف يمكن فهم هذه الآية من منظور أصول الفقه؟",  # How can this verse be understood from an Usul al-Fiqh perspective?
        f"ما هي القواعد الأصولية التي استخدمها العلماء في تفسير هذه الآية؟",  # What Usul al-Fiqh principles did scholars use in interpreting this verse?
        
        # Contemporary scholarly debates
        f"كيف يختلف العلماء المعاصرون في تفسير وتطبيق هذه الآية؟",  # How do contemporary scholars differ in interpreting and applying this verse?
        f"ما هي الآراء المختلفة للمجامع الفقهية المعاصرة حول تطبيق هذه الآية؟",  # What are the different views of contemporary fiqh councils on applying this verse?
        
        # Specific legal implications
        f"ما هي الأحكام الفقهية المستنبطة من هذه الآية في المذاهب المختلفة؟",  # What are the derived legal rulings from this verse in different madhabs?
        f"كيف أثرت الاختلافات في تفسير هذه الآية على الأحكام الفقهية؟"  # How have differences in interpreting this verse affected legal rulings?
    ]
    
    for q in madhab_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "madhab_interpretation"
        })
        
    # More specific controversial topics
    specific_controversial_questions = [
        # Modern scientific issues
        f"كيف تتناول هذه الآية القضايا العلمية مثل نظرية التطور والخلق؟",  # How does this verse address scientific issues like evolution and creation?
        f"ما علاقة هذه الآية بالاكتشافات العلمية الحديثة؟",  # What is the relationship between this verse and modern scientific discoveries?
        
        # Contemporary social issues
        f"كيف يمكن فهم هذه الآية في سياق حقوق المرأة المعاصرة؟",  # How can this verse be understood in the context of contemporary women's rights?
        f"كيف تتناول هذه الآية قضايا التعددية الدينية والتعايش السلمي؟",  # How does this verse address religious pluralism and peaceful coexistence?
        
        # Economic and financial issues
        f"كيف تطبق هذه الآية على النظام المالي المعاصر والمعاملات الحديثة؟",  # How does this verse apply to contemporary financial systems and modern transactions?
        f"ما هو موقف هذه الآية من القضايا الاقتصادية المعاصرة مثل العملات الرقمية؟",  # What is this verse's stance on contemporary economic issues like digital currencies?
        
        # Bioethical questions
        f"كيف تساعدنا هذه الآية في فهم القضايا الطبية الأخلاقية المعاصرة؟",  # How does this verse help us understand contemporary bioethical issues?
        f"ما هو موقف هذه الآية من التقنيات الطبية الحديثة والهندسة الوراثية؟",  # What is this verse's position on modern medical technologies and genetic engineering?
        
        # Environmental issues
        f"كيف تتناول هذه الآية مسؤوليتنا تجاه البيئة والتغير المناخي؟",  # How does this verse address our responsibility towards the environment and climate change?
        f"ما هو منظور هذه الآية حول الاستدامة البيئية واستغلال الموارد الطبيعية؟"  # What is this verse's perspective on environmental sustainability and natural resource exploitation?
    ]
    
    for q in specific_controversial_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "specific_contemporary_issues"
        })
    
    # Linguistic and rhetorical analysis
    linguistic_questions = [
        f"ما هي الأساليب البلاغية المستخدمة في هذه الآية؟",  # What are the rhetorical devices used in this verse?
        f"كيف يؤثر اختيار الكلمات في هذه الآية على المعنى؟",  # How does the choice of words in this verse affect its meaning?
        f"ما هو السياق اللغوي والنحوي لهذه الآية؟"  # What is the linguistic and grammatical context of this verse?
    ]
    
    for q in linguistic_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "linguistic_analysis"
        })

    # Historical context and circumstances of revelation
    historical_questions = [
        f"ما هو سبب نزول هذه الآية؟",  # What was the occasion of revelation for this verse?
        f"ما هو السياق التاريخي الذي نزلت فيه هذه الآية؟",  # What was the historical context in which this verse was revealed?
        f"كيف ارتبطت هذه الآية بأحداث السيرة النبوية؟"  # How was this verse related to events in the Prophet's biography?
    ]
    
    for q in historical_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "historical_context"
        })

    # Thematic connections
    thematic_questions = [
        f"ما هي علاقة هذه الآية بالآيات السابقة واللاحقة؟",  # What is the relationship of this verse to the preceding and following verses?
        f"كيف ترتبط هذه الآية بالمواضيع الرئيسية في السورة؟",  # How does this verse relate to the main themes of the surah?
        f"ما هي الآيات الأخرى في القرآن التي ترتبط بمعنى هذه الآية؟"  # What other verses in the Quran are related to the meaning of this verse?
    ]
    
    for q in thematic_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "thematic_connection"
        })

    # Psychological and spiritual impact
    spiritual_questions = [
        f"كيف تؤثر هذه الآية على الحالة النفسية والروحية للمؤمن؟",  # How does this verse affect the psychological and spiritual state of the believer?
        f"ما هي الجوانب التربوية والروحية في هذه الآية؟",  # What are the educational and spiritual aspects of this verse?
        f"كيف تساهم هذه الآية في تزكية النفس وتهذيبها؟"  # How does this verse contribute to self-purification and refinement?
    ]
    
    for q in spiritual_questions:
        qa_pairs.append({
            "question": q,
            "context": context,
            "answer_text": verse,
            "answer_start": context.find(verse),
            "verse": verse,
            "translation": translation,
            "surah": surah,
            "ayah": ayah,
            "question_type": "spiritual_impact"
        })
    
    # Filter out any QA pairs where answer wasn't found in context
    return [qa for qa in qa_pairs if qa["answer_start"] != -1]

def prepare_qa_dataset(
    input_file: str = "data/quran.csv",
    output_dir: str = "data/qa",
    context_window: int = 5  # Number of verses before and after for context
) -> Tuple[Dataset, Dataset]:
    """
    Prepare the Quran dataset for question answering
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the Quran dataset
    df = pd.read_csv(input_file)
    
    # Initialize lists for QA pairs
    qa_examples = []
    
    # Generate QA pairs for each verse
    for idx in tqdm(range(len(df)), desc="Generating QA pairs"):
        verse = df.iloc[idx]["text"]
        translation = df.iloc[idx]["translation"]
        surah = df.iloc[idx]["surah"]
        ayah = df.iloc[idx]["ayah"]
        
        # Get context (surrounding verses)
        start_idx = max(0, idx - context_window)
        end_idx = min(len(df), idx + context_window + 1)
        context_verses = df.iloc[start_idx:end_idx]["text"].tolist()
        context = " ".join(context_verses)
        
        # Preprocess Arabic text
        verse = preprocess_arabic(verse)
        context = preprocess_arabic(context)
        
        # Generate QA pairs
        qa_pairs = generate_qa_pairs(verse, translation, context, surah, ayah)
        qa_examples.extend(qa_pairs)
    
    print(f"Generated {len(qa_examples)} question-answer pairs")
    
    # Convert to datasets
    dataset = Dataset.from_list(qa_examples)
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Save datasets
    dataset["train"].to_json(os.path.join(output_dir, "train.json"))
    dataset["test"].to_json(os.path.join(output_dir, "test.json"))
    
    return dataset["train"], dataset["test"]

if __name__ == "__main__":
    train_dataset, test_dataset = prepare_qa_dataset() 
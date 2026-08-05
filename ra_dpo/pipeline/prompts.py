"""
Research-backed prompt strategies for sexism detection.

5 strategies based on top NLP papers:
- basic: Simple task description (baseline)
- definition: Full EXIST taxonomy with sexism subcategories
- cot: Chain-of-thought reasoning before classification
- persona: Expert annotator role-playing
- structured: Explicit checklist-based decision framework

References:
- EXIST 2023 annotation guidelines (sexism taxonomy)
- Wei et al. (NeurIPS 2022): Chain-of-thought prompting
- Kirk et al. (NeurIPS 2023): Role-playing in content classification
- Chiu et al. (EMNLP 2022): Prompt-based sexism detection
"""

from typing import Dict, List, Optional


STRATEGIES: Dict[str, Dict[str, Dict[str, str]]] = {
    "basic": {
        "en": {
            "system": "You are a classifier that detects sexist content.",
            "user": (
                "Classify whether this social media post is sexist or not.\n\n"
                "Post: {text}\n\n"
                "Answer (YES or NO):"
            ),
        },
        "es": {
            "system": "Eres un clasificador que detecta contenido sexista.",
            "user": (
                "Clasifica si esta publicacion de redes sociales es sexista o no.\n\n"
                "Post: {text}\n\n"
                "Respuesta (SI o NO):"
            ),
        },
    },
    "definition": {
        "en": {
            "system": (
                "You are an expert content moderator trained in identifying sexist language. "
                "Sexism includes any of the following categories:\n"
                "1. IDEOLOGICAL AND INEQUALITY: Discourses that justify gender inequality or "
                "traditional gender roles as natural or desirable.\n"
                "2. STEREOTYPING AND DOMINANCE: Statements that assert male superiority, "
                "enforce gender stereotypes, or express dominance over women.\n"
                "3. OBJECTIFICATION: Treating women as objects, focusing on physical "
                "appearance in a dehumanizing way, or reducing their value to attractiveness.\n"
                "4. SEXUAL VIOLENCE: References to sexual violence, harassment, threats, "
                "or coercion.\n"
                "5. MISOGYNY AND NON-SEXUAL VIOLENCE: Expressions of hatred towards women, "
                "threats of non-sexual violence, or derogatory language.\n"
                "Respond with ONLY 'YES' if the text contains any form of sexism, or 'NO' if it does not."
            ),
            "user": (
                "Classify whether this social media post contains sexist content.\n\n"
                "Post: {text}\n\n"
                "Classification (YES or NO):"
            ),
        },
        "es": {
            "system": (
                "Eres un moderador de contenido experto en identificar lenguaje sexista. "
                "El sexismo incluye cualquiera de las siguientes categorias:\n"
                "1. IDEOLOGICA Y DESIGUALDAD: Discursos que justifican la desigualdad de genero o "
                "los roles de genero tradicionales como naturales o deseables.\n"
                "2. ESTEREOTIPOS Y DOMINACION: Afirmaciones de superioridad masculina, "
                "refuerzo de estereotipos de genero o expresiones de dominacion sobre las mujeres.\n"
                "3. COSIFICACION: Tratar a las mujeres como objetos, enfocarse en la apariencia "
                "fisica de manera deshumanizante o reducir su valor a su atractivo.\n"
                "4. VIOLENCIA SEXUAL: Referencias a violencia sexual, acoso, amenazas "
                "o coercion.\n"
                "5. MISOGINIA Y VIOLENCIA NO SEXUAL: Expresiones de odio hacia las mujeres, "
                "amenazas de violencia no sexual o lenguaje despectivo.\n"
                "Responde SOLO con 'SI' si el texto contiene alguna forma de sexismo, o 'NO' si no."
            ),
            "user": (
                "Clasifica si esta publicacion de redes sociales contiene contenido sexista.\n\n"
                "Post: {text}\n\n"
                "Clasificacion (SI o NO):"
            ),
        },
    },
    "cot": {
        "en": {
            "system": (
                "You are an expert in detecting sexist content. "
                "For each post, briefly analyze whether it contains sexism, then give your classification. "
                "End your response with EXACTLY 'CLASSIFICATION: YES' or 'CLASSIFICATION: NO'."
            ),
            "user": (
                "Analyze whether this social media post is sexist.\n\n"
                "Post: {text}\n\n"
                "Think step by step:\n"
                "1. Does this post reference gender stereotypes, objectification, sexual violence, "
                "or expressions of gender-based hatred?\n"
                "2. Is the language derogatory, demeaning, or discriminatory towards a gender?\n"
                "3. Could a reasonable person consider this post sexist?\n\n"
                "CLASSIFICATION (YES or NO):"
            ),
        },
        "es": {
            "system": (
                "Eres un experto en detectar contenido sexista. "
                "Para cada publicacion, analiza brevemente si contiene sexismo y luego da tu clasificacion. "
                "Termina tu respuesta con EXACTAMENTE 'CLASIFICACION: SI' o 'CLASIFICACION: NO'."
            ),
            "user": (
                "Analiza si esta publicacion de redes sociales es sexista.\n\n"
                "Post: {text}\n\n"
                "Piensa paso a paso:\n"
                "1. Esta publicacion hace referencia a estereotipos de genero, cosificacion, "
                "violencia sexual o expresiones de odio de genero?\n"
                "2. El lenguaje es despectivo, degradante o discriminatorio hacia un genero?\n"
                "3. Podria una persona razonable considerar esta publicacion sexista?\n\n"
                "CLASIFICACION (SI o NO):"
            ),
        },
    },
    "persona": {
        "en": {
            "system": (
                "You are a trained annotator for the EXIST 2023 shared task on sexism "
                "detection. You have extensive experience identifying sexism in social media "
                "posts in both English and Spanish. You apply the EXIST annotation guidelines "
                "which define sexism as any expression that is demeaning, stereotyping, "
                "objectifying, or expresses violence or hatred based on gender. "
                "You know that subtle sexism (benevolent sexism, microaggressions) is also sexism. "
                "Respond with ONLY 'YES' or 'NO'."
            ),
            "user": (
                "As an expert EXIST annotator, classify this post:\n\n"
                "Post: {text}\n\n"
                "Annotation (YES or NO):"
            ),
        },
        "es": {
            "system": (
                "Eres un anotador entrenado para la tarea compartida EXIST 2023 sobre deteccion "
                "de sexismo. Tienes amplia experiencia identificando sexismo en publicaciones de "
                "redes sociales en ingles y espanol. Aplicas las guias de anotacion EXIST que "
                "definen el sexismo como cualquier expresion degradante, estereotipada, "
                "cosificadora, o que exprese violencia u odio basado en genero. "
                "Sabes que el sexismo sutil (sexismo benevolente, microagresiones) tambien es sexismo. "
                "Responde SOLO con 'SI' o 'NO'."
            ),
            "user": (
                "Como anotador experto de EXIST, clasifica esta publicacion:\n\n"
                "Post: {text}\n\n"
                "Anotacion (SI o NO):"
            ),
        },
    },
    "structured": {
        "en": {
            "system": (
                "You are a sexism detection system. Evaluate the post against the given criteria "
                "and respond with ONLY 'YES' if ANY criterion is met, or 'NO' if NONE are met."
            ),
            "user": (
                "Evaluate this post for sexism:\n\n"
                "Post: {text}\n\n"
                "Criteria (ANY = sexist):\n"
                "- Contains gender stereotypes or sexist ideology\n"
                "- Objectifies or sexualizes based on gender\n"
                "- Uses derogatory gender-based language\n"
                "- Expresses dominance, hatred, or violence based on gender\n"
                "- Contains subtle/benevolent sexism (e.g., 'women belong in the kitchen')\n\n"
                "Classification (YES or NO):"
            ),
        },
        "es": {
            "system": (
                "Eres un sistema de deteccion de sexismo. Evalua la publicacion contra los criterios "
                "dados y responde SOLO con 'SI' si CUALQUIER criterio se cumple, o 'NO' si NINGUNO se cumple."
            ),
            "user": (
                "Evalua esta publicacion por sexismo:\n\n"
                "Post: {text}\n\n"
                "Criterios (CUALQUIERA = sexista):\n"
                "- Contiene estereotipos de genero o ideologia sexista\n"
                "- Cosifica o sexualiza basandose en genero\n"
                "- Usa lenguaje despectivo basado en genero\n"
                "- Expresa dominacion, odio o violencia basada en genero\n"
                "- Contiene sexismo sutil/benevolente\n\n"
                "Clasificacion (SI o NO):"
            ),
        },
    },
}

# Max tokens for each strategy (CoT needs more for reasoning)
STRATEGY_MAX_TOKENS = {
    "basic": 10,
    "definition": 10,
    "cot": 200,
    "persona": 10,
    "structured": 10,
}


class PromptBuilder:
    """Builds prompts for different strategies, languages, and scenarios."""

    def get_system_prompt(self, strategy: str, lang: str) -> str:
        return STRATEGIES[strategy][lang]["system"]

    def get_user_template(self, strategy: str, lang: str) -> str:
        return STRATEGIES[strategy][lang]["user"]

    def get_max_tokens(self, strategy: str) -> int:
        return STRATEGY_MAX_TOKENS.get(strategy, 10)

    def format_user_prompt(
        self,
        text: str,
        lang: str,
        strategy: str,
        examples: Optional[List[Dict]] = None,
    ) -> str:
        """Format user prompt with optional few-shot examples."""
        template = self.get_user_template(strategy, lang)

        if examples:
            examples_str = self._format_examples(examples, lang)
            return f"Here are some examples:\n\n{examples_str}\n\nNow classify:\n\n{template.format(text=text)}"

        return template.format(text=text)

    def format_local_prompt(
        self,
        text: str,
        lang: str,
        strategy: str,
        examples: Optional[List[Dict]] = None,
    ) -> str:
        """Format for local models (combine system + user into single prompt)."""
        system = self.get_system_prompt(strategy, lang)
        user = self.format_user_prompt(text, lang, strategy, examples)
        return f"{system}\n\n{user}"

    def _format_examples(self, examples: List[Dict], lang: str) -> str:
        lines = []
        for i, ex in enumerate(examples, 1):
            label = ex["label"]
            if lang == "es":
                label = "SI" if label == "YES" else "NO"
            lines.append(f"Example {i}:\nPost: {ex['text']}\nClassification: {label}")
        return "\n\n".join(lines)

    @staticmethod
    def parse_prediction(text: str, lang: str = "en") -> str:
        """Parse model output to YES/NO (robust to CoT and varied outputs)."""
        text_upper = text.strip().upper()

        # Check for explicit classification markers (CoT)
        for marker in ["CLASSIFICATION:", "CLASIFICACION:"]:
            if marker in text_upper:
                after = text_upper.split(marker)[-1].strip()
                for token in after.split():
                    clean = token.strip(".:,")
                    if clean in ("YES", "SI"):
                        return "YES"
                    if clean == "NO":
                        return "NO"

        # Direct first-token match
        first_token = text_upper.split()[0].strip(".:,") if text_upper.split() else ""
        if first_token in ("YES", "SI"):
            return "YES"
        if first_token == "NO":
            return "NO"

        # Fallback: scan for last YES/NO
        for token in reversed(text_upper.split()):
            clean = token.strip(".:,")
            if clean in ("YES", "SI"):
                return "YES"
            if clean == "NO":
                return "NO"

        return "NO"

// ===== NLG SYSTEM (EXACT COPY FROM CONTENT.JS) =====
// Natural Language Generation for Deepfake Detection Results

class DeepfakeNLG {
  constructor() {
    // Linguistic templates with slots for contextual filling
    this.templates = {
      deepfake: {
        high_confidence: [
          {
            structure: "[DETERMINATION] [EVIDENCE] [ACTION]",
            determination: [
              "The analysis strongly indicates",
              "VeriFeed has identified clear signs that",
              "Evidence suggests",
              "Multiple indicators show that"
            ],
            evidence: [
              "this video has been manipulated using artificial intelligence.",
              "deepfake technology was used to create this content.",
              "AI-generated alterations are present in this video.",
              "this content was synthetically modified using AI tools."
            ],
            action: [
              "Users should verify this through other sources before sharing.",
              "Fact-checking is recommended before distributing this content.",
              "Cross-referencing with original sources is advised before sharing.",
              "Verification through trusted sources is recommended before considering it authentic."
            ]
          }
        ],
        medium_confidence: [
          {
            structure: "[POSSIBILITY] [EVIDENCE] [CAUTION]",
            possibility: [
              "This video may have been",
              "There are indicators suggesting this was",
              "The system has detected signs that this could be",
              "Analysis indicates this might have been"
            ],
            evidence: [
              "edited or created using artificial intelligence.",
              "manipulated with deepfake technology.",
              "generated using AI tools.",
              "altered using synthetic media techniques."
            ],
            caution: [
              "Verification is recommended before sharing.",
              "Additional verification is advised.",
              "Caution should be exercised when sharing this content.",
              "Fact-checking through multiple sources is recommended before sharing."
            ]
          }
        ],
        low_confidence: [
          {
            structure: "[UNCERTAINTY] [OBSERVATION] [RECOMMENDATION]",
            uncertainty: [
              "While certainty is limited,",
              "The analysis is inconclusive, but",
              "VeriFeed has limited confidence that",
              "Though uncertain,"
            ],
            observation: [
              "this video shows some signs of manipulation.",
              "there may be AI-generated elements present.",
              "artificial alterations might be present.",
              "possible synthetic modifications were detected."
            ],
            recommendation: [
              "This content should be treated with skepticism and verified through multiple sources.",
              "Fact-checking is strongly recommended before sharing.",
              "Additional analysis is needed before drawing conclusions.",
              "Additional verification methods should be consulted before trusting this content."
            ]
          }
        ]
      },
      authentic: {
        high_confidence: [
          {
            structure: "[DETERMINATION] [EVIDENCE] [ASSESSMENT]",
            determination: [
              "The analysis indicates",
              "VeriFeed has found strong evidence that",
              "Multiple factors suggest",
              "The assessment shows"
            ],
            evidence: [
              "this video is genuine and has not been digitally manipulated.",
              "this content appears authentic with no signs of AI generation.",
              "this video shows no indicators of deepfake technology.",
              "this is authentic content without synthetic alterations."
            ],
            assessment: [
              "However, important content should always be verified through trusted sources.",
              "Still, cross-referencing with original sources is good practice.",
              "Verification through official channels is still recommended when possible.",
              "As always, critical content should be verified through additional sources."
            ]
          }
        ],
        medium_confidence: [
          {
            structure: "[LIKELIHOOD] [EVIDENCE] [CAUTION]",
            likelihood: [
              "This video appears to be",
              "VeriFeed believes this is likely",
              "Evidence suggests this is probably",
              "Analysis indicates this is most likely"
            ],
            evidence: [
              "authentic and unmanipulated.",
              "genuine with no AI alterations.",
              "real content without deepfake elements.",
              "legitimate with no synthetic modifications."
            ],
            caution: [
              "though verification is recommended for complete certainty.",
              "but additional verification is always recommended.",
              "though exercising caution is still advisable.",
              "however, verification through trusted sources is advised when important."
            ]
          }
        ],
        low_confidence: [
          {
            structure: "[UNCERTAINTY] [OBSERVATION] [RECOMMENDATION]",
            uncertainty: [
              "The system cannot confidently determine",
              "The analysis is inconclusive about",
              "VeriFeed has low confidence in assessing",
              "It's unclear from the analysis"
            ],
            observation: [
              "whether this video is authentic or manipulated.",
              "if this content contains AI-generated elements.",
              "the authenticity of this video.",
              "whether synthetic alterations are present."
            ],
            recommendation: [
              "Verification through multiple trusted sources is recommended before relying on this content.",
              "This should be treated with caution until verified.",
              "Additional expert analysis may be needed for confirmation.",
              "Verification from authoritative sources is advised before trusting this content."
            ]
          }
        ]
      }
    };
  }

  /**
   * NLP Text Generation Function
   * Uses linguistic rules and context to generate varied messages
   * Implements slot-filling algorithm for template-based NLG
   */
  generate(prediction, confidence) {
    const isAuthentic = prediction === "REAL";
    const category = isAuthentic ? "authentic" : "deepfake";
    
    // Determine confidence level using linguistic thresholds
    let confidenceLevel;
    if (confidence >= 80) {
      confidenceLevel = "high_confidence";
    } else if (confidence >= 60) {
      confidenceLevel = "medium_confidence";
    } else {
      confidenceLevel = "low_confidence";
    }
    
    // Get appropriate template set
    const templateSet = this.templates[category][confidenceLevel];
    
    // Select template (randomized for linguistic variation)
    const template = templateSet[Math.floor(Math.random() * templateSet.length)];
    
    // Generate sentence parts using slot-filling / regex (NLG technique)
    const parts = template.structure.match(/\[([^\]]+)\]/g).map(slot => {
      const slotName = slot.replace(/[\[\]]/g, '').toLowerCase();
      const options = template[slotName];
      // Random selection provides linguistic variety
      return options[Math.floor(Math.random() * options.length)];
    });
    
    // Construct final sentence with proper spacing
    return parts.join(' ');
  }

  generateConfidenceText(confidence) {
    if (confidence >= 90) {
      const options = [
        "VeriFeed has very high confidence in this assessment",
        "The analysis provides very strong certainty",
        "The system is highly confident in this determination",
        "This assessment has very high reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else if (confidence >= 80) {
      const options = [
        "VeriFeed has high confidence in this assessment",
        "The analysis provides strong certainty",
        "The system is confident in this determination",
        "This assessment has high reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else if (confidence >= 70) {
      const options = [
        "VeriFeed has moderate confidence in this assessment",
        "The analysis suggests reasonable certainty",
        "The system is moderately confident in this determination",
        "This assessment has moderate reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else if (confidence >= 60) {
      const options = [
        "VeriFeed has limited confidence in this assessment",
        "The analysis suggests some uncertainty",
        "The system is somewhat confident in this determination",
        "This assessment has limited reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else {
      const options = [
        "VeriFeed has low confidence in this assessment",
        "The analysis is highly uncertain",
        "The system has minimal confidence in this determination",
        "This assessment has low reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    }
  }

  /**
  * Grammar correction utility (NLP function)
  */
  correctGrammar(text) {
    return text
      .replace(/\s+/g, ' ')
      .replace(/\s([.,!?])/g, '$1')
      .replace(/^./, str => str.toUpperCase())
      .trim();
  }
}

// Export for use in popup.js
const deepfakeNLG = new DeepfakeNLG();

console.log('[VeriFeed NLG] Natural Language Generation module loaded');
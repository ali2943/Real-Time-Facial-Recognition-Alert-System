"""
Intelligent Decision Engine
Smart decision making with multi-factor scoring - FIXED VERSION
"""

import numpy as np


class IntelligentDecisionEngine:
    """
    Multi-factor decision engine
    
    Combines:
    1. Distance score (from face matching)
    2. Quality score (from quality checker)
    3. Liveness score (from liveness detector)
    4. Temporal consistency (from tracking)
    
    Makes intelligent decision instead of binary pass/fail
    """
    
    def __init__(self):
        self.decision_threshold = 0.55  # Lowered from 0.60
        print("[INFO] Intelligent Decision Engine initialized")
    
    def make_decision(self, match_result, quality_score=1.0, liveness_score=1.0, 
                     temporal_confidence=1.0):
        """
        Make intelligent access decision
        
        Args:
            match_result: (user, distance, adaptive_threshold)
            quality_score: Face quality (0-100, will be normalized)
            liveness_score: Liveness confidence (0-1)
            temporal_confidence: Tracking confidence (0-1)
            
        Returns:
            (decision, overall_score, details)
            decision: 'GRANT', 'DENY', 'MFA_REQUIRED'
        """
        user, distance, threshold = match_result
        
        if user is None:
            return 'DENY', 0.0, {'reason': 'Unknown person', 'distance': distance}
        
        # Calculate component scores
        scores = {}
        
        # ================================================
        # 1. Distance score - FIXED WITH BETTER FORMULA
        # ================================================
        # Lower distance = higher score
        # Uses square root for gentle curve that rewards good matches
        if distance <= threshold:
            # Normalize distance to 0-1 range
            normalized = distance / threshold
            
            # Square root formula - gives better score distribution
            # Examples:
            #   dist=0.0  → score=1.00 (100%)
            #   dist=0.25 → score=0.87 (87%)
            #   dist=0.50 → score=0.71 (71%)
            #   dist=0.75 → score=0.50 (50%)
            #   dist=1.00 → score=0.00 (0%)
            distance_score = (1.0 - normalized) ** 0.5
            
            # Clamp to valid range
            distance_score = max(0.0, min(1.0, distance_score))
        else:
            # Distance exceeds threshold - no match
            distance_score = 0.0
        
        scores['distance'] = distance_score
        
        # 2. Quality score (normalize from 0-100 to 0-1)
        quality_normalized = min(1.0, quality_score / 100.0)
        scores['quality'] = quality_normalized
        
        # 3. Liveness score (already 0-1)
        scores['liveness'] = liveness_score
        
        # 4. Temporal consistency score
        scores['temporal'] = temporal_confidence
        
        # Calculate weighted overall score
        weights = {
            'distance': 0.50,    # Most important - actual face match
            'quality': 0.15,     # Quality of image
            'liveness': 0.25,    # Anti-spoofing
            'temporal': 0.10     # Consistency over time
        }
        
        overall_score = sum(scores[k] * weights[k] for k in scores)
        
        # ================================================
        # Decision logic - ADJUSTED THRESHOLDS
        # ================================================
        
        if overall_score >= 0.75:
            # High confidence - grant access immediately
            decision = 'GRANT'
            reason = f"High confidence match ({overall_score:.1%})"
            
        elif overall_score >= 0.55:  # LOWERED from 0.60
            # Medium-high confidence
            # Check critical factors
            if scores['liveness'] < 0.3:  # LOWERED from 0.4
                decision = 'DENY'
                reason = "Failed liveness check (possible spoofing)"
            elif scores['quality'] < 0.25:  # LOWERED from 0.3
                decision = 'DENY'
                reason = "Image quality too low"
            else:
                decision = 'GRANT'
                reason = f"Acceptable match ({overall_score:.1%})"
                
        elif overall_score >= 0.40:  # LOWERED from 0.45
            # Medium-low confidence - borderline
            if scores['distance'] >= 0.35 and scores['liveness'] >= 0.35:  # LOWERED from 0.5
                # Face matches reasonably and seems live
                decision = 'GRANT'
                reason = f"Borderline acceptable ({overall_score:.1%})"
            else:
                decision = 'DENY'
                reason = f"Insufficient confidence ({overall_score:.1%})"
                
        else:
            # Low confidence - deny
            decision = 'DENY'
            reason = f"Low confidence ({overall_score:.1%})"
        
        details = {
            'overall_score': overall_score,
            'component_scores': scores,
            'weights': weights,
            'distance': distance,
            'threshold': threshold,
            'user': user,
            'reason': reason
        }
        
        return decision, overall_score, details
    
    def explain_decision(self, details):
        """
        Generate human-readable explanation
        
        For debugging and transparency
        """
        lines = []
        lines.append(f"╔{'═'*50}╗")
        lines.append(f"║ DECISION ANALYSIS{' '*32}║")
        lines.append(f"╠{'═'*50}╣")
        lines.append(f"║ User: {details['user']:<43}║")
        lines.append(f"║ Overall Score: {details['overall_score']:.1%}{' '*35}║")
        lines.append(f"║ Distance: {details['distance']:.4f} / {details['threshold']:.4f}{' '*26}║")
        lines.append(f"║ Reason: {details['reason']:<42}║")
        lines.append(f"╠{'═'*50}╣")
        lines.append(f"║ COMPONENT BREAKDOWN{' '*31}║")
        lines.append(f"╠{'═'*50}╣")
        
        for component, score in details['component_scores'].items():
            weight = details['weights'][component]
            contribution = score * weight
            bar_length = int(score * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            lines.append(f"║ {component.capitalize():12} {score:.1%} {bar} {contribution:.1%}  ║")
        
        lines.append(f"╚{'═'*50}╝")
        
        return "\n".join(lines)


# Test/Demo
if __name__ == "__main__":
    print("="*60)
    print("INTELLIGENT DECISION ENGINE - TEST")
    print("="*60)
    
    engine = IntelligentDecisionEngine()
    
    # Test case 1: Your actual data
    print("\n[TEST 1] Your Actual Case:")
    print("Distance: 0.3351, Threshold: 0.4000")
    
    decision, score, details = engine.make_decision(
        match_result=("Ali", 0.3351, 0.4000),
        quality_score=100.0,
        liveness_score=1.0,
        temporal_confidence=1.0
    )
    
    print(f"\n{engine.explain_decision(details)}")
    print(f"\nDecision: {decision}")
    
    # Calculate expected scores
    normalized = 0.3351 / 0.4000  # 0.8377
    distance_score = (1.0 - normalized) ** 0.5  # sqrt(0.1623) = 0.4029 = 40.29%
    overall = 0.4029 * 0.5 + 1.0 * 0.15 + 1.0 * 0.25 + 1.0 * 0.10
    
    print(f"\nExpected Calculation:")
    print(f"  Normalized distance: {normalized:.4f}")
    print(f"  Distance score: {distance_score:.1%}")
    print(f"  Overall: {overall:.1%}")
    
    # Test case 2: Perfect match
    print("\n" + "="*60)
    print("[TEST 2] Perfect Match:")
    print("Distance: 0.1000, Threshold: 0.4000")
    
    decision2, score2, details2 = engine.make_decision(
        match_result=("TestUser", 0.1000, 0.4000),
        quality_score=100.0,
        liveness_score=1.0,
        temporal_confidence=1.0
    )
    
    print(f"\n{engine.explain_decision(details2)}")
    print(f"\nDecision: {decision2}")
    
    # Test case 3: Borderline
    print("\n" + "="*60)
    print("[TEST 3] Borderline Match:")
    print("Distance: 0.3800, Threshold: 0.4000")
    
    decision3, score3, details3 = engine.make_decision(
        match_result=("TestUser", 0.3800, 0.4000),
        quality_score=80.0,
        liveness_score=0.5,
        temporal_confidence=1.0
    )
    
    print(f"\n{engine.explain_decision(details3)}")
    print(f"\nDecision: {decision3}")
    
    print("\n" + "="*60)
    print("FORMULA COMPARISON")
    print("="*60)
    
    test_distances = [0.1, 0.2, 0.3, 0.35, 0.4]
    threshold = 0.4
    
    print(f"\nThreshold: {threshold}")
    print(f"{'Distance':<10} {'Linear':<10} {'Sqrt':<10} {'Improvement'}")
    print("-"*50)
    
    for dist in test_distances:
        norm = dist / threshold
        linear = 1.0 - norm
        sqrt = (1.0 - norm) ** 0.5
        improvement = ((sqrt - linear) / linear * 100) if linear > 0 else 0
        
        print(f"{dist:<10.2f} {linear:<10.1%} {sqrt:<10.1%} +{improvement:.1f}%")
import numpy as np

# รายชื่อคลาสของโมเดล 3 คลาส
CLASS_NAMES_3CLS = {
    0: "Normal (เหตุการณ์ปกติ)",
    1: "Property_Crime (ภัยต่อทรัพย์สิน)",
    2: "Personal_Violence (ความรุนแรงต่อบุคคล)"
}

def filter_prediction(probabilities, threshold=0.70):
    """
    ฟังก์ชันกรองความมั่นใจ (Confidence Threshold Filter)
    
    หลักการทำงาน:
    - หาคลาสที่โมเดลทายว่ามีความน่าจะเป็นสูงสุด (argmax)
    - หากคลาสที่ได้คือ Property_Crime (1) หรือ Personal_Violence (2) แต่ความน่าจะเป็น (Confidence) ต่ำกว่า threshold
      ระบบจะปัดผลลัพธ์กลับไปเป็น Normal (0) ทันทีเพื่อป้องกันการเตือนภัยมั่ว (False Alarm)
    """
    probabilities = np.array(probabilities)
    pred_class = np.argmax(probabilities)
    confidence = probabilities[pred_class]
    
    is_filtered = False
    final_class = pred_class
    
    # หากทำนายคลาสอันตราย (1 หรือ 2) แต่ความมั่นใจต่ำกว่าเกณฑ์
    if pred_class in [1, 2] and confidence < threshold:
        final_class = 0
        is_filtered = True
        
    return {
        "raw_class": pred_class,
        "raw_label": CLASS_NAMES_3CLS[pred_class],
        "final_class": final_class,
        "final_label": CLASS_NAMES_3CLS[final_class],
        "confidence": confidence,
        "is_filtered": is_filtered
    }

def main():
    print("==========================================================")
    print("   Demo: Action Classifier Confidence Threshold Filter    ")
    print("==========================================================")
    print("วัตถุประสงค์: แสดงผลลัพธ์การแก้ไขปัญหาสัญญาณเตือนภัยมั่ว (False Positives)")
    print("ค่ากำหนดระดับความมั่นใจขั้นต่ำ (Threshold) = 70%\n")

    # จำลองสถานการณ์น่าจะเป็น (Mock Probabilities) จากเอาต์พุตโมเดล
    scenarios = [
        {
            "desc": "กรณีที่ 1: คนร้ายเดินเข้ามางัดตู้นิรภัยตรงๆ (ชัดเจน)",
            "probs": [0.05, 0.92, 0.03]  # Property_Crime เด่นชัดเจน
        },
        {
            "desc": "กรณีที่ 2: พนักงานยืนปัดกวาดพื้นแถวลิ้นชัก (โมเดลสับสนทายว่ากำลังโจรกรรมแบบมั่นใจต่ำ)",
            "probs": [0.35, 0.55, 0.10]  # Property_Crime นำแต่ไม่มั่นใจ
        },
        {
            "desc": "กรณีที่ 3: เกิดเหตุคนยกหมัดชกต่อยปะทะกัน (ชัดเจน)",
            "probs": [0.02, 0.10, 0.88]  # Personal_Violence เด่นชัดเจน
        },
        {
            "desc": "กรณีที่ 4: คนเดินสวนและโบกมือทักทายกันแรง (โมเดลสับสนทายว่าเป็นความรุนแรงแบบมั่นใจต่ำ)",
            "probs": [0.40, 0.15, 0.45]  # Personal_Violence นำแต่ไม่มั่นใจ
        },
        {
            "desc": "กรณีที่ 5: คนนั่งทำงานพิมพ์งานปกติ",
            "probs": [0.95, 0.03, 0.02]  # Normal นำชัดเจน
        }
    ]

    for i, sc in enumerate(scenarios, 1):
        print(f"🎬 {sc['desc']}")
        print(f"   [เวกเตอร์ความมั่นใจจากโมเดล]: Normal={sc['probs'][0]:.2f}, Property_Crime={sc['probs'][1]:.2f}, Violence={sc['probs'][2]:.2f}")
        
        # รันฟังก์ชันกรอง
        result = filter_prediction(sc["probs"], threshold=0.70)
        
        print(f"   🔴 ก่อนกรอง (Raw ArgMax): {result['raw_label']} (ความมั่นใจ {result['confidence']*100:.1f}%)")
        
        if result['is_filtered']:
            print(f"   ✅ หลังกรอง (Thresholded): {result['final_label']} <-- [กรองเปลี่ยนผลลัพธ์สำเร็จ!]")
        else:
            print(f"   📢 หลังกรอง (Thresholded): {result['final_label']} <-- [ส่งสัญญาณไซเรนปกติ]")
        print("-" * 58)

    print("\n[บทสรุปสำหรับนำเสนออาจารย์]")
    print("1. สัญญาณเตือนภัยมั่ว (False Positives) ส่วนใหญ่ในระบบ CCTV เกิดจากพฤติกรรมปกติที่มีความเร็วขยับตัวใกล้เคียงกับเหตุร้าย")
    print("2. การใช้เกณฑ์ความมั่นใจ (Threshold) 70% จะสกัดสัญญาณที่ไม่มั่นใจและปัดกลับเป็น 'เหตุการณ์ปกติ'")
    print("3. ทำให้เจ้าหน้าที่ในห้องควบคุมระบบไม่รำคาญจากสัญญาณเตือนภัยมั่ว และเพิ่มความเชื่อมั่นให้กับระบบความปลอดภัยจริง")
    print("==========================================================")

if __name__ == "__main__":
    main()

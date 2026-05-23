from metrics import MUART244

lasher = MUART244(
    gt_path=f'eval_tracker-ua/metrics/gt_file/MUART244/annos/visible',
    seq_name_path=f"eval_tracker-ua/metrics/gt_file/MUART244/testinglist.txt")


name = '/data1/Code/jinjiandong/Neuro_Unaligned-0925/eval_tracker/tracking_result_muart244/AFter'

lasher(
    tracker_name="tracker2",
    result_path=f"{name}/MUART244", 
    bbox_type="ltwh")


# Evaluate multiple trackers
pr_dict = lasher.PR()
npr_dict = lasher.NPR()
sr_dict = lasher.SR()

# print(pr_dict["tracker1"][0])
# print(npr_dict["tracker1"][0])
# print(sr_dict["tracker1"][0])

print(pr_dict["tracker2"][0])
print(npr_dict["tracker2"][0])
print(sr_dict["tracker2"][0])

# lasher.draw_plot(metric_fun=lasher.PR)
# lasher.draw_plot(metric_fun=lasher.NPR)
# lasher.draw_plot(metric_fun=lasher.SR)
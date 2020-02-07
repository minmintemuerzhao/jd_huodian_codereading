import numpy as np


# 有功功率
def add_power(df, window=7 * 24 * 60 * 3):
    coal_feed_name = ['A给煤机瞬时给煤量反馈', 'B给煤机瞬时给煤量反馈', 'C给煤机瞬时给煤量反馈', 'D给煤机瞬时给煤量反馈', 'E给煤机瞬时给煤量反馈', 'F给煤机瞬时给煤量反馈']
    coal_current_name = ['A磨煤机电流', 'B磨煤机电流', 'C磨煤机电流', 'D磨煤机电流', 'E磨煤机电流', 'F磨煤机电流']
    for i in range(len(coal_feed_name)):
        coal_feed = df[coal_feed_name[i]].values
        coal_current = df[coal_current_name[i]].values
        power = coal_feed / coal_current
        name = '单耗' + chr(65 + i)
        index = np.where(coal_current > 15)[0]
        df[name] = power
        mean = df[name][index].mean()
        index = np.where(coal_current < 15)[0]
        df[name][index] = mean
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        tmp = df[name].values
        # 使用numpy卷积函数来实现滑窗平均值
        df[name] = np.convolve(tmp, np.ones((window,)) / window, mode='same')


# 额外给煤量
def add_extra_coal(df):
    load_name = '机组实际负荷'
    coal_feed_name = ['A给煤机瞬时给煤量反馈', 'B给煤机瞬时给煤量反馈', 'C给煤机瞬时给煤量反馈', 'D给煤机瞬时给煤量反馈', 'E给煤机瞬时给煤量反馈', 'F给煤机瞬时给煤量反馈']
    for i in range(len(coal_feed_name)):
        if i == 0:
            sum = df[coal_feed_name[i]].values
        else:
            sum = sum+df[coal_feed_name[i]].values
    x = df[load_name]
    y = sum
    poly = np.polyfit(x, y, deg=1)

    cal_coal = poly[0] * x + poly[1]
    extra_coal = y - cal_coal
    df['额外给煤量'] = extra_coal
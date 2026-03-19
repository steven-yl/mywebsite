---
title: "draft"
subtitle: ""
date: 2026-03-06T00:00:00+08:00
draft: true
authors: [Steven]
description: ""

tags: [draft]
categories: [draft]
series: [draft系列]
weight: 0
series_weight: 0

hiddenFromHomePage: false
hiddenFromSearch: false
---


{{< echarts >}}
title:
    text: 折线统计图
    top: 2%
    left: center
tooltip:
    trigger: axis
legend:
    data:
        - 邮件营销
        - 联盟广告
        - 视频广告
        - 直接访问
        - 搜索引擎
    top: 10%
grid:
    left: 5%
    right: 5%
    bottom: 5%
    top: 20%
    containLabel: true
toolbox:
    feature:
        saveAsImage:
            title: 保存为图片
xAxis:
    type: category
    boundaryGap: false
    data:
        - 周一
        - 周二
        - 周三
        - 周四
        - 周五
        - 周六
        - 周日
yAxis:
    type: value
series:
    - name: 邮件营销
      type: line
      stack: 总量
      data:
          - 120
          - 132
          - 101
          - 134
          - 90
          - 230
          - 210
    - name: 联盟广告
      type: line
      stack: 总量
      data:
          - 220
          - 182
          - 191
          - 234
          - 290
          - 330
          - 310
    - name: 视频广告
      type: line
      stack: 总量
      data:
          - 150
          - 232
          - 201
          - 154
          - 190
          - 330
          - 410
    - name: 直接访问
      type: line
      stack: 总量
      data:
          - 320
          - 332
          - 301
          - 334
          - 390
          - 330
          - 320
    - name: 搜索引擎
      type: line
      stack: 总量
      data:
          - 820
          - 932
          - 901
          - 934
          - 1290
          - 1330
          - 1320
{{< /echarts >}}
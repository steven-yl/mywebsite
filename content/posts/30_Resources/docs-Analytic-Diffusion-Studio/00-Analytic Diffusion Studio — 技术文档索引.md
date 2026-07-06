---
title: "Analytic Diffusion Studio — 技术文档索引"
date: 2026-05-30T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 技术文档索引：Obsidian 双链导航。"
summary: "本目录全部笔记的 wikilink 索引，便于图谱与反向链接浏览。"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, docs Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 0
series_weight: 0
hiddenFromHomePage: false
seriesNavigation: false
hiddenFromSearch: false
---

> 本文档对 Analytic Diffusion Studio 项目进行全面、深入的技术解读。文档按"总体→局部"组织，先给出整体架构总览，再按模块逐章展开。

## 文档目录

| 笔记 | 说明 |
|------|------|
| [[01-Analytic Diffusion Studio — 项目总览]] | 项目总览：背景知识、整体架构、模块关系、方法对比 |
| [[02-Analytic Diffusion Studio — 扩散模型理论基础]] | 扩散模型理论基础：前向/反向过程、DDIM 调度器、去噪公式 |
| [[03-Analytic Diffusion Studio — 配置系统]] | 配置系统：OmegaConf 层级合并、数据类定义、路径解析、CLI 覆盖 |
| [[04-Analytic Diffusion Studio — 数据模块]] | 数据模块：注册机制、数据集工厂、预处理管线、支持的数据集 |
| [[05-Analytic Diffusion Studio — 模型基类与采样循环]] | 模型基类：BaseDenoiser 接口、DDIM 采样循环、轨迹记录 |
| [[06-Analytic Diffusion Studio — Wiener 滤波去噪器]] | Wiener 滤波去噪器：协方差矩阵、SVD 分解、线性去噪公式 |
| [[07-Analytic Diffusion Studio — 最优贝叶斯去噪器]] | 最优贝叶斯去噪器：FAISS 索引、softmax 加权平均、温度参数 |
| [[08-Analytic Diffusion Studio — 平滑最优去噪器]] | 平滑最优去噪器 (SCFDM)：高斯扰动平均、继承关系 |
| [[09-Analytic Diffusion Studio — PCA Locality 去噪器]] | PCA Locality 去噪器：局部性掩码、流式 softmax、核心创新 |
| [[10-Analytic Diffusion Studio — 最近邻基线]] | 最近邻基线：欧氏距离检索 |
| [[11-Analytic Diffusion Studio — 基线 UNet 模型]] | 基线 UNet：神经网络架构、权重加载、对比评估 |
| [[12-Analytic Diffusion Studio — 评估指标与实验流程]] | 评估指标与实验流程：R²、MSE、WandB 集成、输出结构 |
| [[13-Analytic Diffusion Studio — 工具模块]] | 工具模块：Wiener 滤波计算/存储、UNet 网络组件 |

## 相关笔记

- [[01-Analytic Diffusion Studio — 项目总览]]
- [[02-Analytic Diffusion Studio — 扩散模型理论基础]]

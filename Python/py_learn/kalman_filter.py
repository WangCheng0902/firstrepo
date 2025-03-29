import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, 
                 transition_matrix, observation_matrix,
                 process_noise, measurement_noise):
        """
        初始化卡尔曼滤波器
        
        参数:
        initial_state: 初始状态向量 (n x 1)
        initial_covariance: 初始协方差矩阵 (n x n)
        transition_matrix: 状态转移矩阵 (n x n) 
        observation_matrix: 观测矩阵 (m x n)
        process_noise: 过程噪声协方差矩阵 (n x n)
        measurement_noise: 测量噪声协方差矩阵 (m x m)
        """
        self.state = initial_state
        self.covariance = initial_covariance
        self.F = transition_matrix
        self.H = observation_matrix
        self.Q = process_noise
        self.R = measurement_noise
        
    def predict(self):
        """预测步骤"""
        # 预测状态
        self.state = self.F @ self.state
        # 预测协方差
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state
    
    def update(self, measurement):
        """更新步骤"""
        # 计算卡尔曼增益
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态估计
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        
        # 更新协方差估计
        I = np.eye(self.covariance.shape[0])
        self.covariance = (I - K @ self.H) @ self.covariance
        
        return self.state

# 示例使用
if __name__ == "__main__":
    # 1D位置和速度跟踪示例
    dt = 1.0  # 时间步长
    
    # 初始化参数
    initial_state = np.array([[0], [0]])  # [位置, 速度]
    initial_covariance = np.eye(2) * 0.1
    transition_matrix = np.array([[1, dt], 
                                 [0, 1]])
    observation_matrix = np.array([[1, 0]])
    process_noise = np.eye(2) * 0.01
    measurement_noise = np.eye(1) * 0.1
    
    # 创建卡尔曼滤波器
    kf = KalmanFilter(initial_state, initial_covariance,
                     transition_matrix, observation_matrix,
                     process_noise, measurement_noise)
    
    # 模拟测量数据
    true_position = 5.0
    measurements = [true_position + np.random.normal(0, 0.5) for _ in range(10)]
    
    # 运行滤波器并收集数据
    print("卡尔曼滤波跟踪结果:")
    measurements_list = []
    estimates_pos = []
    estimates_vel = []
    
    for z in measurements:
        kf.predict()
        estimated_state = kf.update(np.array([[z]]))
        print(f"测量值: {z:.2f}, 估计位置: {estimated_state[0,0]:.2f}, 估计速度: {estimated_state[1,0]:.2f}")
        measurements_list.append(z)
        estimates_pos.append(estimated_state[0,0])
        estimates_vel.append(estimated_state[1,0])
    
    # 绘制结果图
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    # 绘制位置跟踪结果
    plt.subplot(2, 1, 1)
    plt.plot(measurements_list, 'r*', label='测量值')
    plt.plot(estimates_pos, 'b-', label='估计位置')
    plt.axhline(y=true_position, color='g', linestyle='--', label='真实位置')
    plt.ylabel('位置')
    plt.title('卡尔曼滤波跟踪结果')
    plt.legend()
    
    # 绘制速度估计结果
    plt.subplot(2, 1, 2)
    plt.plot(estimates_vel, 'm-', label='估计速度')
    plt.axhline(y=0, color='g', linestyle='--', label='真实速度')
    plt.xlabel('时间步')
    plt.ylabel('速度')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kalman_filter_results.png')  # 保存为图片文件
    print("滤波结果图已保存为: kalman_filter_results.png")

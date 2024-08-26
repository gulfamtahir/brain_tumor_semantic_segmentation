####### I need to change the Shape of the data ################
new_shape = (212, 3, 512, 512)
X_train = np.transpose(X_train, axes=(0, 3, 1, 2))
y_train = np.transpose(y_train, axes=(0, 3, 1, 2))
X_val= np.transpose(X_train, axes=(0, 3, 1, 2))
y_val= np.transpose(X_train, axes=(0, 3, 1, 2))


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Assuming binary mask (0,1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor )


train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True )
val_data_loader = DataLoader(val_dataset, batch_size=8, shuffle=False )

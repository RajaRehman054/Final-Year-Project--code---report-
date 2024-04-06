import React, { useContext } from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

import TabNavigator from "../screens/TabScreens/TabNavigator";
import Notification from "../screens/StackScreens/Notifications";
import MessageChat from "../screens/StackScreens/MessageChat";
import DirectMessage from "../screens/StackScreens/DirectMessage";
import Settings from "../screens/StackScreens/Settings";
import OthersProfile from "../screens/StackScreens/OthersProfile";
import RecentSearch from "../screens/StackScreens/RecentSearch";
import Teleprompter from "../screens/StackScreens/Teleprompter";
import CameraRecording from "../screens/StackScreens/CameraRecording";
import PostContentScreen from "../screens/StackScreens/PostContentScreen";
import BuyContent from "../screens/StackScreens/BuyContent";
import Jobdetails from "../screens/StackScreens/Jobdetails";
import { UserContext } from "../context/UserContext";
import Loader from "../components/Loader";

const Stack = createNativeStackNavigator();

const MainStack = () => {
  const { loading } = useContext(UserContext);

  if (loading) {
    return <Loader />;
  }

  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="TabNavigator" component={TabNavigator} />
      <Stack.Screen name="Notification" component={Notification} />
      <Stack.Screen name="MessageChat" component={MessageChat} />
      <Stack.Screen name="DirectMessage" component={DirectMessage} />
      <Stack.Screen name="Settings" component={Settings} />
      <Stack.Screen name="OthersProfile" component={OthersProfile} />
      <Stack.Screen name="RecentSearch" component={RecentSearch} />
      <Stack.Screen name="CameraRecording" component={CameraRecording} />
      <Stack.Screen name="Teleprompter" component={Teleprompter} />
      <Stack.Screen name="PostContentScreen" component={PostContentScreen} />
      <Stack.Screen name="BuyContent" component={BuyContent} />
      <Stack.Screen name="Jobdetails" component={Jobdetails} />
    </Stack.Navigator>
  );
};

export default MainStack;

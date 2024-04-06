import React, { useState } from "react";
import { payment } from "../../api/Api";

const PaymentModal = () => {
  const [price, setPrice] = useState(1);
  const AddMoneyHandler = async () => {
    const res = await payment(price);
    window.location.href = res.url;
  };

  return (
    <dialog id="my_modal_3" className="modal ">
      <div className="modal-box w-[400px] bg-white ">
        <form method="dialog">
          {/* if there is a button in form, it will close the modal */}
          <button className="btn btn-circle btn-ghost btn-sm absolute right-2 top-2">
            ✕
          </button>
        </form>
        <div>
          <h3 className="text-center text-lg font-bold text-[#FF8216]">
            Add Money to wallet
          </h3>
          <div className="mt-[30px] flex w-full flex-row items-center">
            <p className=" w-[50%] text-[20px] font-bold">Enter an amount</p>

            <input
              className="h-[60px] w-[50%] rounded-[10px] border border-[#FF8216] bg-[#FF8216] bg-opacity-10 px-[30px] shadow-input focus:border-heading focus:outline-none active:outline-none"
              placeholder="$"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </div>
          <div className="mt-[20px] flex w-full">
            <button
              className="h-[60px] w-full rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]"
              style={{ transition: "0.3s" }}
              onClick={AddMoneyHandler}
            >
              Go
            </button>
          </div>
        </div>
      </div>
    </dialog>
  );
};

export default PaymentModal;

//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright 2016 Qualcomm Technologies, Inc. All rights reserved.
//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
//
//  The party receiving this software directly from QTI (the "Recipient")
//  may use this software as reasonably necessary solely for the purposes
//  set forth in the agreement between the Recipient and QTI (the
//  "Agreement"). The software may be used in source code form solely by
//  the Recipient's employees (if any) authorized by the Agreement. Unless
//  expressly authorized in the Agreement, the Recipient may not sublicense,
//  assign, transfer or otherwise provide the source code to any third
//  party. Qualcomm Technologies, Inc. retains all ownership rights in and
//  to the software
//
//  This notice supersedes any other QTI notices contained within the software
//  except copyright notices indicating different years of publication for
//  different portions of the software. This notice does not supersede the
//  application of any third party copyright notice to that third party's
//  code.
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================

#ifndef _DL_SYSTEM_OPTIONAL_HPP_
#define _DL_SYSTEM_OPTIONAL_HPP_
#include <cstdio>
#include <utility>

#include "DlSystem/ZdlExportDefine.hpp"

namespace zdl {
namespace DlSystem {

template <typename T>

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief .
 *
 * Class to manage a value that may or may not exist. The boolean value
 * of the Optional class is true if the object contains a value and false
 * if it does not contain a value.
 *
 * The class must be evaluated and confirmed as true (containing a value)
 * before being dereferenced.
 */
class ZDL_EXPORT Optional final {
public:
    /**
    * The default constructor is set to not have any value, and is
     * therefore evaluated as false.
    */
   // Do not explicit it so we can return {}
   Optional() = default;

    /**
    * Construct an Optional class using an object.
     * @param[in] Reference to an object v
     * @param[out] Optional instance of object v
    */
   Optional (const T& v) : m_Init(true) {
      try {
         new (static_cast<void*>(m_Storage)) T(v);
      } catch (...) {
         // Let the Optional<> instance be valid with init==false
         m_Init = false;
      }
   }
   Optional(const Optional &other) : m_Init(other.m_Init) {
      if (m_Init) {
         try {
            new (static_cast<void*>(m_Storage)) T(other.GetReference());
         } catch (...) {
            // Let the Optional<> instance be valid with init==false
            m_Init = false;
         }
      }
   }
   Optional(Optional &&other) noexcept {
      this->swap(std::move(other));
   }
   ~Optional() {
      if (m_Init) {
         this->GetReference().~T();
      }
      m_Init = false;
   }

    /**
    * Boolean value of Optional class is only true when there exists a value.
    */
   operator bool() const noexcept { return m_Init; }

   Optional& operator=(Optional other) noexcept {
      this->swap(std::move(other));
      return *this;
   }
    // Users should be checking validity otherwise this is junk
    // throw exception in just in case.
   const T& operator*() const {
      if (!m_Init) {
         throw std::bad_exception();
      }
      return this->GetReference();
   }
   const T& operator*() {
      return (static_cast<const zdl::DlSystem::Optional<T>*>(this))->operator*();
   }

   T operator->() {
      if (!m_Init) {
         throw std::bad_exception();
      }
      T self = this->GetReference();
      return self;
   }
   operator T&() {
      if (!m_Init) {
         throw std::bad_exception();
      }
      return this->GetReference();
   }
 private:
   void swap(Optional &&other) noexcept {
      std::swap(m_Init, other.m_Init);
      std::swap(m_Storage, other.m_Storage);
   }
    /**
    * Get reference of Optional object
    * @warning User must validate Optional has value before.
    */
   const T& GetReference() const noexcept {
      return *static_cast<const T*>(reinterpret_cast<const void*>(m_Storage)); }
    T& GetReference() noexcept {
       return *static_cast<T*>(reinterpret_cast<void*>(m_Storage)); }
   T GetPointer()  noexcept {
      return static_cast<T>(reinterpret_cast<void*>(m_Storage)); }

   alignas(T) char m_Storage[sizeof(T)];
   bool m_Init = false;
};

} // ns DlSystem
} // ns zdl

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // _DL_SYSTEM_OPTIONAL_HPP_
